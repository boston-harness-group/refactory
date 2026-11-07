"""Run a data node's items concurrently with a parallelism bound.

When ``node.parallelism > 1`` and the serial-fallback checks pass, the walking
executor dispatches here instead of the sequential ``for`` loop.  Each item
gets its own ``Executor`` + ``RunContext`` clone via the shared helpers in
``concurrent_helpers.py`` (factored from ``fork.py``), and a
``ThreadPoolExecutor`` bounded by ``parallelism`` runs them.

The contract is the sequential loop's, item for item:

- the step cap is the one the sequential loop has, scaled by the item count;
- the per-item deadline covers each item's setup and its verify, not its walk
  (``Executor._data_item_deadline``; the spine's per-command timeout). A phase
  that outlives it is abandoned: a thread cannot be stopped, so its alias is
  removed and later writes through that path fail. That stops writes made by
  path through the workspace it was given; it does not stop a subprocess it
  started with that folder as its cwd, or a path it had already resolved, which
  keep writing into the pass (the sequential alarm interrupts the phase instead);
- a halt or a gate stop ends the node and is recorded as the sequential loop
  records it (``Executor._record``), a ``BudgetExhausted`` propagates, and the
  records of the items that finished are kept;
- an item that a halt, a gate stop or a budget stop cancelled is neither
  recorded nor marked finished, so a resume runs it again;
- records, ``item_scores`` and the merged steps are in item order, whatever
  order the items finished in;
- a resume skips the items whose pass finished (``.done``), as the sequential
  loop's resume does.
"""
from __future__ import annotations

import copy
import json
import sys
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .concurrent_helpers import clone_child_executor, make_step_budget_reserve, merge_children_into_parent

if TYPE_CHECKING:
    from .executor import Executor
    from .graph import Node


@dataclass
class _ItemResult:
    """Outcome of one data-node item."""
    index: int
    record: dict[str, Any] | None = None
    #: "halted" or "refused" when the item stopped the node, as the sequential loop stops.
    stop: str | None = None
    error: str | None = None
    #: A halt, gate stop or budget stop elsewhere ended this item before it finished.
    cancelled: bool = False
    child: Any = None


@dataclass
class ConcurrentOutcome:
    """What the concurrent item loop hands back to ``Executor.run``."""
    scores: list[dict[str, Any]] = field(default_factory=list)
    #: "halted" or "refused" when an item stopped the node; None otherwise.
    stop: str | None = None
    error: str | None = None
    #: Step indices (relative to the data loop's first step) of failed items.
    failed_step_indices: set[int] = field(default_factory=set)


def is_shareable_model(model: Any) -> bool:
    """Whether one model client may serve several items at once.

    A client says so with a ``thread_safe = True`` attribute (``DshModelClient``
    does: each completion is its own subprocess). Anything else is not assumed
    safe, because a client that keeps per-call state would mix items' answers.
    """
    return model is None or getattr(model, "thread_safe", False) is True


def _fallback_reasons(executor: "Executor", node: "Node") -> list[str]:
    from refactory_dsh.memory import MemoryStore

    ctx = executor.ctx
    reasons: list[str] = []
    memory = ctx.memory
    if memory is not None:
        if type(memory) is not MemoryStore:
            reasons.append(
                f"memory {type(memory).__name__} is not a MemoryStore, the only store "
                "known safe to share across items")
        else:
            declared = sorted(memory.describe())
            if declared:
                reasons.append(
                    f"the MemoryStore declares namespace(s) {', '.join(declared)}, whose "
                    "reads and writes are not synchronised across concurrent items")
    if not is_shareable_model(ctx.model):
        reasons.append(
            f"model client {type(ctx.model).__name__} does not declare thread_safe = True, "
            "so it is not shared across concurrent items")
    for nid in executor.graph.data_body(node.id):
        body_node = executor.graph.nodes.get(nid)
        if (body_node is not None
                and body_node.kind == "gate"
                and body_node.gate_kind != "fn"
                and not ctx.verdicts.get(body_node.id)):
            reasons.append(f"gate {nid!r} needs a person's answer")
    return reasons


def _check_serial_fallback(
    executor: "Executor",
    node: "Node",
    parallelism: int,
) -> dict[str, Any] | None:
    """Return the recorded fallback fact if the node must run serially, or None.

    A fallback is recorded as the ``data_parallelism_effective`` fact and a
    ``run_warning`` event, so no run claims parallelism it did not get. A warning
    that cannot be written is reported on stderr and in the fact, never dropped.
    """
    reasons = _fallback_reasons(executor, node)
    if not reasons:
        return None
    ctx = executor.ctx
    reason_text = "; ".join(reasons)
    fact: dict[str, Any] = {
        "node": node.id,
        "requested": parallelism,
        "effective": 1,
        "reason": reason_text,
    }
    try:
        from refactory_dsh.live import emit
        emit(
            ctx.workdir, "run_warning",
            node_id=node.id, state="warning",
            summary=f"data parallelism fallback to serial: {reason_text}",
        )
    except Exception as exc:  # noqa: BLE001 - reported below, not swallowed
        fact["warning_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"warning": "data_parallelism_fallback", **fact}), file=sys.stderr)
    ctx._internal_fact("data_parallelism_effective", fact)  # noqa: SLF001
    return fact


def _within_deadline(fn: Callable[[], Any], deadline_s: float | None, node_id: str, item_id: str) -> Any:
    """Run ``fn`` under the per-item deadline the sequential loop's alarm enforces.

    The sequential loop arms ``signal.alarm(int(max(1, deadline)))`` around setup
    and again around verify. A worker thread cannot take SIGALRM, so the phase runs
    on a helper thread and the item fails with the same timeout error when the
    phase is still running at the deadline.
    """
    from .executor import _ItemDeadlineError

    if deadline_s is None:
        return fn()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the item's thread
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True, name=f"{node_id}:{item_id}")
    worker.start()
    worker.join(int(max(1, deadline_s)))
    if worker.is_alive():
        raise _ItemDeadlineError(node_id, item_id, deadline_s)
    if "error" in box:
        raise box["error"]
    return box.get("value")


#: Marks a pass whose setup or verify outlived the item deadline; it holds the phase.
TIMED_OUT = ".timed_out"


def _live_alias(run_workdir: Path, node_id: str, pass_dir: Path) -> Path:
    """A symlink to *pass_dir* that one item works through, under ``.refactory/live``.

    Whatever is at the alias path is replaced: a dangling link from a stopped run,
    or a real folder an abandoned phase re-created there (``mkdir(parents=True)``,
    ``shutil.copytree``) after its alias was removed.
    """
    import shutil

    from refactory_dsh.items import live_dir

    alias = live_dir(run_workdir, node_id) / pass_dir.name
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink() or alias.is_file():
        alias.unlink()
    elif alias.exists():
        shutil.rmtree(alias)
    alias.symlink_to(pass_dir.resolve(), target_is_directory=True)
    return alias


def _metadata_fields(item: dict[str, Any], instance: Any) -> dict[str, Any]:
    meta = instance.metadata
    return {
        "metadata": {k: v for k, v in item.items() if k not in ("id", "path")},
        "label": str(meta.get("label", "")) if meta.get("label") is not None else None,
        "label_text": str(meta.get("label_text", "")) if meta.get("label_text") is not None else None,
    }


def run_data_concurrent(
    executor: "Executor",
    node: "Node",
    items: list[dict[str, Any]],
    task: Any,
    parallelism: int,
    *,
    setup_kwargs: dict[str, Any] | None = None,
    verify_kwargs: dict[str, Any] | None = None,
    item_deadline_s: float | None = None,
    item_gate: threading.BoundedSemaphore | None = None,
) -> ConcurrentOutcome:
    """Run data-node items concurrently, with the sequential loop's semantics.

    The finished items' records are appended to ``executor._item_records`` in item
    order before this returns or re-raises, as the sequential loop has appended
    them by the time it stops.
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
        pass_read_through,
        prepare_pass_dir,
    )
    from refactory_dsh.task import instance_of

    passthrough = (BudgetExhausted, KeyboardInterrupt, SystemExit)
    run_workdir = executor._run_workdir or executor.ctx.workdir  # noqa: SLF001
    cancelled = threading.Event()
    # The cap the sequential loop walks under: max_steps already scaled by the
    # item count (Executor.run), less the steps taken so far.
    reserve = make_step_budget_reserve(executor, cancelled)
    base_model = executor.ctx.model
    declared_reads = pass_read_through(
        {n.id: (n.reads, n.writes) for n in executor.graph.nodes.values()},
        executor.graph.data_body(node.id),
        node.writes,
    )
    setup_kwargs = setup_kwargs or {}
    verify_kwargs = verify_kwargs or {}

    def item_model() -> Any:
        bind = getattr(base_model, "with_cancellation", None)
        return bind(cancelled) if callable(bind) else base_model

    def run_item(index: int) -> _ItemResult:
        item = items[index]
        item_id = str(item.get("id") or index)
        result = _ItemResult(index=index)

        done_dir = pass_dir_of(run_workdir, node.id, item_id, index)
        if (done_dir / ".done").exists():
            result.record = finished_pass_record(done_dir)
            return result
        if cancelled.is_set():
            result.cancelled = True
            return result

        # Acquire run-wide item gate (cross-evaluation limiter).
        # The acquire is immediately before the try/finally so that any
        # exception in setup code (instance_of, prepare_pass_dir,
        # clone_child_executor, etc.) still releases the permit, preventing
        # deadlock under disk-full or memory pressure.
        _has_gate = item_gate is not None
        if _has_gate:
            item_gate.acquire()
        try:

            instance = instance_of(item)
            pass_dir = prepare_pass_dir(run_workdir, node.id, item_id, index, declared_reads=declared_reads)
            child = clone_child_executor(executor, item_model(), reserve_step=reserve, gate_iterations={})
            # The item works through an alias of its pass folder, removed when the item
            # ends -- at once when a setup or verify outlives its deadline, so the
            # abandoned phase's later writes *through that path* fail. A subprocess it
            # started in the folder, or a path it resolved, is not stopped (see the
            # module docstring); the sequential loop's alarm interrupts the phase instead.
            live = _live_alias(run_workdir, node.id, pass_dir) if item_deadline_s is not None else None
            child.ctx.workdir = live or pass_dir
            phase = "setup"
            child._run_workdir = run_workdir  # noqa: SLF001
            result.child = child
            item_task = copy.deepcopy(task) if task is not None else None
            fields = _metadata_fields(item, instance)
            record: dict[str, Any] | None = None
            finished = False

            try:
                if item_task is not None:
                    phase = "setup"
                    _within_deadline(
                        lambda: item_task.setup(instance, child.ctx.workdir, **setup_kwargs),
                        item_deadline_s, node.id, item_id,
                    )
                child.ctx.node_id = node.id
                child.ctx._declared = node.writes  # noqa: SLF001
                child.ctx._declared_reads = node.reads  # noqa: SLF001
                child.ctx._declared_facts = node.produces  # noqa: SLF001
                child.ctx._produced = set()  # noqa: SLF001
                from refactory_dsh.task import node_item

                # The nodes read the item without its grader-only fields.
                child.ctx.write_json(CURRENT_ITEM, node_item(item, task, child.ctx.knobs))
                child._item_exchanges = []  # noqa: SLF001
                branch_error = child._walk_branch(  # noqa: SLF001
                    node.subgraph_entry, node.subgraph_exit,
                    include_barrier=True, iterations={},
                )
                if branch_error is not None:
                    raise _BranchFailed(branch_error)
                phase = "verify"
                verdict = (
                    _within_deadline(
                        lambda: item_task.verify(instance, child.ctx.workdir, **verify_kwargs),
                        item_deadline_s, node.id, item_id,
                    )
                    if item_task is not None else None
                )
            except passthrough:
                cancelled.set()
                raise
            except HaltStop as exc:
                result.stop, result.error = "halted", str(exc)
                cancelled.set()
                return result
            except GateStop as exc:
                result.stop, result.error = "refused", str(exc)
                cancelled.set()
                return result
            except CancelledError:
                result.cancelled = True
                return result
            except _ItemDeadlineError as exc:
                if live is not None:
                    live.unlink(missing_ok=True)
                (pass_dir / TIMED_OUT).write_text(phase)
                record = make_item_record(item_id, None, error=exc,
                                          exchanges=list(child._item_exchanges), **fields)  # noqa: SLF001
                if record is not None:
                    record["error_kind"] = "timeout"
                finished = True
            except _BranchFailed as exc:
                if cancelled.is_set():
                    # Stopped by another item's halt, gate or budget stop, not failed.
                    result.cancelled = True
                    return result
                record = make_item_record(item_id, None, error=RuntimeError(str(exc)),
                                          exchanges=list(child._item_exchanges),  # noqa: SLF001
                                          fail_reason=str(exc), **fields)
                finished = True
            except Exception as exc:  # noqa: BLE001 - a failed item, as in the sequential loop
                if cancelled.is_set():
                    result.cancelled = True
                    return result
                record = make_item_record(item_id, None, error=exc,
                                          exchanges=list(child._item_exchanges), **fields)  # noqa: SLF001
                finished = True
            else:
                prediction: str | None = None
                if verdict is not None:
                    if verdict.prediction:
                        prediction = verdict.prediction
                    elif verdict.passed:
                        prediction = verdict.details
                    else:
                        prediction = ""
                record = make_item_record(
                    item_id, verdict,
                    exchanges=list(child._item_exchanges),  # noqa: SLF001
                    prediction=prediction,
                    fail_reason=verdict.details if verdict is not None and not verdict.passed and verdict.details else None,
                    **fields,
                )
                finished = True
            finally:
                if live is not None:
                    live.unlink(missing_ok=True)
                if finished:
                    mark_pass_complete(pass_dir, record)
            result.record = record
            return result

        finally:
            if _has_gate:
                item_gate.release()

    results: list[_ItemResult] = [_ItemResult(index=i) for i in range(len(items))]
    raised: BaseException | None = None
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(run_item, i): i for i in range(len(items))}
        try:
            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                except BaseException as exc:  # noqa: BLE001 - re-raised after the merge
                    cancelled.set()
                    if raised is None:
                        raised = exc
        except BaseException:
            # Interrupted on this thread: stop scheduling, let the running items stop.
            cancelled.set()
            for future in futures:
                future.cancel()
            raise

    # The aliases went with their items; an empty folder is not left behind. One an
    # abandoned phase wrote into stays until the node is next reset.
    from refactory_dsh.items import live_dir
    for leftover in (live_dir(run_workdir, node.id), live_dir(run_workdir, node.id).parent):
        try:
            leftover.rmdir()
        except OSError:
            break
    outcome = ConcurrentOutcome()
    merge_children_into_parent(executor, [r.child for r in results if r.child is not None])
    step_base = 0
    allowed = set(node.produces) - {"item_scores"}
    for r in results:
        if r.child is not None:
            steps = len(r.child.steps)
            if r.record is not None and r.record.get("error_text") is not None:
                outcome.failed_step_indices.update(range(step_base, step_base + steps))
            step_base += steps
            # The declared facts each finished pass changed, folded in item order
            # (the sequential loop's pass_scope merge).
            if not r.cancelled and r.stop is None:
                for key in allowed & set(r.child.ctx.facts):
                    if executor.ctx.facts.get(key) != r.child.ctx.facts[key]:
                        executor.ctx.facts[key] = r.child.ctx.facts[key]
        if r.record is not None:
            outcome.scores.append(r.record)
            executor._item_records.append(r.record)  # noqa: SLF001
        if r.stop is not None and outcome.stop is None:
            outcome.stop, outcome.error = r.stop, r.error
    if raised is not None:
        raise raised
    return outcome
