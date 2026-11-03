"""Run a data node's items concurrently with a parallelism bound.

When ``node.parallelism > 1`` and the serial-fallback checks pass, the walking
executor dispatches here instead of the sequential ``for`` loop.  Each item
gets its own ``Executor`` + ``RunContext`` clone via the shared helpers in
``concurrent_helpers.py`` (factored from ``fork.py``), and a
``ThreadPoolExecutor`` bounded by ``parallelism`` runs them.

Concurrency safety:
- ``current_item.json`` is written to the pass directory only (not run-level).
- ``facts.json`` and ``pass_state`` writes go through file-locked helpers.
- ``item_scores`` are accumulated per-child and merged in item order after all
  items complete.
- Per-item deadlines use ``threading.Timer`` (not SIGALRM) in concurrent mode.
- Budget is shared and thread-safe (``RunBudget`` uses ``fcntl``).
"""
from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .concurrent_helpers import clone_child_executor, make_step_budget_reserve

if TYPE_CHECKING:
    from .executor import Executor
    from .graph import Node


@dataclass
class _ItemResult:
    """Outcome of one data-node item."""
    index: int
    item_id: str
    record: dict[str, Any] | None = None
    error: str | None = None
    halted: bool = False
    gate_stopped: bool = False
    child: Any = None
    failed_step_indices: list[int] = field(default_factory=list)


def _is_thread_safe_model(model: Any) -> bool:
    """Whether the model client is known to be thread-safe.

    Accepts the bare ``DshModelClient`` *and* wrapped/rebound copies
    (``for_node``, ``with_budget``, ``with_wall``, ``with_cancellation``,
    ``with_tool_policy``).  All of these are ``copy.copy`` clones of the
    original that keep ``type(x) is DshModelClient`` true, or are
    subclasses that maintain the same subprocess-per-call thread-safety
    contract.
    """
    from refactory_dsh.model import DshModelClient

    if model is None:
        return True
    return isinstance(model, DshModelClient)


def _check_serial_fallback(
    executor: "Executor",
    node: "Node",
    parallelism: int,
) -> dict[str, Any] | None:
    """Return a fallback reason dict if we must run serially, or None.

    When a fallback fires, the reason is recorded as a warning and a fact
    so no run silently claims parallelism it did not get (correction 1).
    """
    ctx = executor.ctx

    from refactory_dsh.memory import MemoryStore

    reasons: list[str] = []

    if ctx.memory is not None and (
        type(ctx.memory) is not MemoryStore or bool(ctx.memory.describe())
    ):
        reasons.append("non-MemoryStore memory is not thread-safe")

    if not _is_thread_safe_model(ctx.model):
        reasons.append(f"model client {type(ctx.model).__name__} is not known thread-safe")

    # Check for user/agent gates in the data body that lack pre-supplied verdicts.
    body_nodes = executor.graph.data_body(node.id)
    for nid in body_nodes:
        body_node = executor.graph.nodes.get(nid)
        if (body_node is not None
                and body_node.kind == "gate"
                and body_node.gate_kind != "fn"
                and not ctx.verdicts.get(body_node.id)):
            reasons.append(f"gate {nid!r} needs a person's answer")

    if not reasons:
        return None

    reason_text = "; ".join(reasons)
    fact = {
        "node": node.id,
        "requested": parallelism,
        "effective": 1,
        "reason": reason_text,
    }
    # Record run warning.
    try:
        from refactory_dsh.live import emit
        emit(
            ctx.workdir, "run_warning",
            node_id=node.id, state="warning",
            summary=f"data parallelism fallback to serial: {reason_text}",
        )
    except Exception:
        pass
    # Record fact.
    ctx._internal_fact("data_parallelism_effective", fact)
    return fact


def run_data_concurrent(
    executor: "Executor",
    node: "Node",
    items: list[dict[str, Any]],
    task: Any,
    parallelism: int,
    *,
    setup_kwargs: dict[str, Any] | None = None,
    verify_kwargs: dict[str, Any] | None = None,
    original_max_steps: int | None = None,
) -> tuple[
    list[dict[str, Any]],   # scores (in item order)
    list[dict[str, Any]],   # item_records (in item order)
    str | None,              # error
    bool,                    # halted
    set[int],                # failed step indices
]:
    """Run data-node items concurrently.

    Returns ``(scores, item_records, error, halted, failed_step_indices)``.
    The caller merges these into the executor's state the same way the
    sequential path does.
    """
    from .executor import (
        CURRENT_ITEM,
        GateStop,
        HaltStop,
        _BranchFailed,
        _ItemDeadlineError,
    )
    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.items import (
        finished_pass_record,
        make_item_record,
        mark_pass_complete,
        pass_dir_of,
        prepare_pass_dir,
    )
    from refactory_dsh.task import instance_of

    _PASSTHROUGH = (BudgetExhausted, KeyboardInterrupt, SystemExit)
    _run_workdir = executor._run_workdir or executor.ctx.workdir

    cancelled = threading.Event()

    # Step budget reserve (shared with fork).
    # Use the original max_steps (before the data-loop scaling) for the budget.
    _budget_remaining: int | None = None
    if original_max_steps is not None:
        _budget_remaining = original_max_steps - len(executor.steps)
    reserve = make_step_budget_reserve(executor, cancelled, remaining=_budget_remaining)

    # Model with cancellation scope.
    model = executor.ctx.model
    if _is_thread_safe_model(model) and model is not None:
        from refactory_dsh.model import DshModelClient
        if isinstance(model, DshModelClient):
            model = model.with_cancellation(cancelled)

    # Declared reads for pass isolation.
    from refactory_dsh.items import pass_read_through
    _declared_reads = pass_read_through(
        {n.id: (n.reads, n.writes) for n in executor.graph.nodes.values()},
        executor.graph.data_body(node.id),
        node.writes,
    )

    # Per-item deadline.
    _item_deadline_s = executor._data_item_deadline()

    setup_kwargs = setup_kwargs or {}
    verify_kwargs = verify_kwargs or {}

    results: list[_ItemResult] = [_ItemResult(index=i, item_id="") for i in range(len(items))]
    lock = threading.Lock()
    _halt_exc: list[Any] = []

    def run_item(index: int) -> _ItemResult:
        item = items[index]
        item_id = str(item.get("id") or index)
        result = _ItemResult(index=index, item_id=item_id)

        # Resume: skip finished items.
        _done_dir = pass_dir_of(_run_workdir, node.id, item_id, index)
        if (_done_dir / ".done").exists():
            kept = finished_pass_record(_done_dir)
            if kept is not None:
                result.record = kept
            return result

        if cancelled.is_set():
            return result

        instance = instance_of(item)

        # Create isolated pass directory.
        _pass_dir = prepare_pass_dir(
            _run_workdir, node.id, item_id, index,
            declared_reads=_declared_reads,
        )

        # Clone child executor for this item.
        child = clone_child_executor(
            executor, model,
            reserve_step=reserve,
            gate_iterations={},  # per-item gate counters
        )
        child.ctx.workdir = _pass_dir
        child.ctx.facts = copy.deepcopy(executor.ctx.facts)
        child._run_workdir = _run_workdir
        result.child = child

        _pass_completed = False
        record_out = None
        _deadline_timer: threading.Timer | None = None
        _deadline_event = threading.Event()

        try:
            # Setup phase.
            if task is not None:
                task.setup(instance, child.ctx.workdir, **setup_kwargs)

            # Write current_item to pass dir only (not run-level for safety).
            child.ctx.node_id = node.id
            child.ctx._declared = node.writes
            child.ctx._declared_reads = node.reads
            child.ctx._declared_facts = node.produces
            child.ctx._produced = set()
            child.ctx.write_json(CURRENT_ITEM, item)
            child._item_exchanges = []

            # Arm per-item deadline via Timer (not SIGALRM).
            if _item_deadline_s is not None:
                def _on_deadline():
                    _deadline_event.set()
                    cancelled_copy = threading.Event()
                    cancelled_copy.set()
                    # Stop model calls for this item.
                    if hasattr(child.ctx.model, '_cancellations'):
                        for c in child.ctx.model._cancellations:
                            c.set()

                _deadline_timer = threading.Timer(_item_deadline_s, _on_deadline)
                _deadline_timer.daemon = True
                _deadline_timer.start()

            # Walk the branch.
            branch_error = child._walk_branch(
                node.subgraph_entry, node.subgraph_exit,
                include_barrier=True, iterations={},
            )

            if _deadline_event.is_set():
                raise _ItemDeadlineError(node.id, item_id, _item_deadline_s)

            if branch_error is not None:
                raise _BranchFailed(branch_error)

            # Verify phase.
            if task is not None:
                verdict = task.verify(instance, child.ctx.workdir, **verify_kwargs)
            else:
                verdict = None

        except _PASSTHROUGH:
            if _deadline_timer is not None:
                _deadline_timer.cancel()
            raise
        except HaltStop as exc:
            result.halted = True
            result.error = str(exc)
            _halt_exc.append(exc)
            cancelled.set()
            if _deadline_timer is not None:
                _deadline_timer.cancel()
            return result
        except GateStop as exc:
            result.gate_stopped = True
            result.error = str(exc)
            cancelled.set()
            if _deadline_timer is not None:
                _deadline_timer.cancel()
            return result
        except _ItemDeadlineError as exc:
            metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
            record_out = make_item_record(
                item_id, None, error=exc,
                exchanges=list(child._item_exchanges),
                label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                metadata=metadata,
            )
            if record_out is not None:
                record_out["error_kind"] = "timeout"
            _pass_completed = True
        except _BranchFailed as exc:
            metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
            record_out = make_item_record(
                item_id, None, error=RuntimeError(str(exc)),
                exchanges=list(child._item_exchanges),
                fail_reason=str(exc),
                label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                metadata=metadata,
            )
            _pass_completed = True
        except Exception as exc:
            metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
            record_out = make_item_record(
                item_id, None, error=exc,
                exchanges=list(child._item_exchanges),
                label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                metadata=metadata,
            )
            _pass_completed = True
        else:
            metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
            _prediction = None
            if verdict is not None:
                if verdict.prediction:
                    _prediction = verdict.prediction
                elif verdict.passed:
                    _prediction = verdict.details
                else:
                    _prediction = ""
            record_out = make_item_record(
                item_id, verdict,
                exchanges=list(child._item_exchanges),
                prediction=_prediction,
                label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                fail_reason=verdict.details if verdict is not None and not verdict.passed and verdict.details else None,
                metadata=metadata,
            )
            _pass_completed = True
        finally:
            if _deadline_timer is not None:
                _deadline_timer.cancel()
            if _pass_completed:
                mark_pass_complete(_pass_dir, record_out)

        result.record = record_out
        return result

    # Submit all items to the thread pool.
    all_results: list[_ItemResult] = [_ItemResult(index=i, item_id="") for i in range(len(items))]
    error_out: str | None = None
    halted_out = False
    failed_step_indices: set[int] = set()

    try:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {}
            try:
                for i in range(len(items)):
                    futures[pool.submit(run_item, i)] = i
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        all_results[idx] = future.result()
                    except BudgetExhausted:
                        cancelled.set()
                        raise
            except BaseException:
                cancelled.set()
                for f in futures:
                    f.cancel()
                raise
    except BudgetExhausted as exc:
        error_out = str(exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        error_out = f"{node.id}: {type(exc).__name__}: {exc}"

    # Merge results in item order.
    scores: list[dict[str, Any]] = []
    item_records: list[dict[str, Any]] = []
    _step_base = len(executor.steps)

    for i in range(len(items)):
        r = all_results[i]
        if r.child is not None:
            step_start = len(executor.steps)
            executor.steps.extend(r.child.steps)
            executor.model_calls += r.child.model_calls
            executor.ctx.writes_made.extend(r.child.ctx.writes_made)
            # Track failed item step indices.
            if r.record is not None and r.record.get("error_text") is not None:
                failed_step_indices.update(range(step_start, len(executor.steps)))
            # Merge declared facts from child.
            allowed = set(node.produces) | {"item_scores"}
            for key in allowed:
                if key in r.child.ctx.facts and key != "item_scores":
                    if key not in executor.ctx.facts or executor.ctx.facts[key] != r.child.ctx.facts[key]:
                        executor.ctx.facts[key] = r.child.ctx.facts[key]
        if r.record is not None:
            scores.append(r.record)
            item_records.append(r.record)
        if r.halted:
            halted_out = True
            if error_out is None:
                error_out = r.error
        if r.gate_stopped:
            if error_out is None:
                error_out = r.error

    if _halt_exc and error_out is None:
        error_out = str(_halt_exc[0])
        halted_out = True

    return scores, item_records, error_out, halted_out, failed_step_indices
