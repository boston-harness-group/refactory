"""Shared helpers for concurrent branch/item execution on the walker.

Factored from ``fork.py`` so both ``run_fork`` and ``run_data_concurrent`` use
the same step-budget reserve, cancellation, child Executor/RunContext cloning
and ordered-merge logic, rather than duplicating it.
"""
from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .executor import Executor, RunContext
    from .graph import Node


def make_step_budget_reserve(
    executor: "Executor",
    cancelled: threading.Event,
) -> Callable[[], bool]:
    """A thread-safe ``reserve()`` closure that deducts one step from the budget.

    The budget is the executor's cap as it stands -- for a data loop, the cap the
    loop already scaled by its item count -- less the steps already taken.
    Returns ``False`` when it is spent; raises ``CancelledError`` once the
    cancellation event is set, so the walker stops dispatching new work.

    When the executor is itself a branch or an item with a reserve (a fork nested in
    a fork branch or in a concurrent data item), each step is also charged to that
    reserve, so nested work cannot overrun the cap the outer walk shares.
    """
    lock = threading.Lock()
    remaining_box: list[int] = [executor.max_steps - len(executor.steps)]
    outer = executor._reserve_step  # noqa: SLF001

    def reserve() -> bool:
        from concurrent.futures import CancelledError

        with lock:
            if cancelled.is_set():
                raise CancelledError("cancelled before the next step")
            if remaining_box[0] <= 0:
                return False
            remaining_box[0] -= 1
        return outer is None or outer()

    return reserve


def clone_child_executor(
    executor: "Executor",
    model: Any,
    *,
    budget: Any = None,
    reserve_step: Callable[[], bool] | None = None,
    gate_iterations: dict[str, int] | None = None,
) -> "Executor":
    """Create an independent child ``Executor`` + ``RunContext`` for one branch or item.

    The child shares the budget and gate iteration counters with the parent
    (both are thread-safe or idempotent-per-branch), but has its own facts,
    steps, and record.
    """
    from .executor import Executor, RunContext
    from .record import RunRecord

    ctx = executor.ctx
    child_ctx = RunContext(
        ctx.workdir,
        knobs=copy.deepcopy(ctx.knobs),
        facts=copy.deepcopy(ctx.facts),
        verdicts=copy.deepcopy(ctx.verdicts),
        task=copy.deepcopy(ctx.task),
        model=model,
        memory=ctx.memory,
        enforce_writes=ctx.enforce_writes,
    )
    child = Executor(
        executor.graph,
        child_ctx,
        max_steps=executor.max_steps,
        enable_recovery=executor.enable_recovery,
        record=RunRecord(feedback=dict(executor.record.feedback)),
    )
    child_ctx.budget = budget if budget is not None else ctx.budget
    child._gate_iterations = gate_iterations if gate_iterations is not None else executor._gate_iterations
    child._reserve_step = reserve_step
    return child


def merge_children_into_parent(
    executor: "Executor",
    children: list["Executor"],
    order: list[int] | None = None,
) -> None:
    """Merge steps, model_calls and writes from children into the parent in order.

    *order* is the list of indices in merge order (default: natural order).
    """
    if order is None:
        order = list(range(len(children)))
    for idx in order:
        child = children[idx]
        executor.steps.extend(child.steps)
        executor.model_calls += child.model_calls
        executor.ctx.writes_made.extend(child.ctx.writes_made)
