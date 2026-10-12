"""Execute a built graph in this process.

One node per branch at a time, with independent fork branches scheduled concurrently
and every control-flow branch decided by a gate's verdict. That model is why the
validator can make promises a free-for-all executor could not: fan-out is forbidden on a non-gate node, every
cycle passes through a gate, and a gate carries the iteration bound that stops
it.

The executor also holds the dataflow model to its word. A node declares what it
writes; an op that writes anything else is refused, because the validator's
"every read is produced by some node" promise is only true if the declarations
are.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .graph import Graph, Node
from .loops import loop_edges
from .mode import RunResult
from .resolve import resolve

if TYPE_CHECKING:
    from refactory_dsh.task import Task


class ExecutionError(RuntimeError):
    """A run could not be carried out."""


class ModelClient(Protocol):
    """The one thing an agent node needs: text in, text out."""

    def complete(self, prompt: str, *, system: str = "") -> str: ...


@dataclass
class Step:
    """One visited node."""

    node: str
    kind: str
    duration_s: float
    verdict: str | None = None
    error: str | None = None


class RunContext:
    """What a node's callable sees: a work directory, the knobs, and the facts.

    Facts are the run's own numbers, and they are what ``score()`` reads. A write
    is checked against the node's declaration so the graph's dataflow statements
    stay true.
    """

    def __init__(
        self,
        workdir: Path,
        *,
        knobs: dict[str, Any] | None = None,
        facts: dict[str, Any] | None = None,
        verdicts: dict[str, str] | None = None,
        model: ModelClient | None = None,
        task: dict[str, Any] | None = None,
        memory: Any = None,
        enforce_writes: bool = True,
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.knobs: dict[str, Any] = dict(knobs or {})
        self.facts: dict[str, Any] = dict(facts or {})
        #: Verdicts for gates a person or an agent answers, by node id. Empty in a
        #: headless run, which is why such a gate refuses rather than deciding.
        self.verdicts: dict[str, str] = dict(verdicts or {})
        self.model = model
        from refactory_dsh.budgets import RunBudget

        self.budget = RunBudget.inherited(model)
        self.task: dict[str, Any] = dict(task or {})
        #: The mode's declared memory, for an op that has something to remember.
        self.memory = memory
        self.enforce_writes = enforce_writes
        self.iteration = 0
        self.node_id = ""
        self._declared: tuple[str, ...] = ()
        self._declared_facts: tuple[str, ...] = ()
        self._produced: set[str] = set()
        self.writes_made: list[str] = []

    # ── knobs ──────────────────────────────────────────────────────────────
    def knob(self, name: str, default: Any = None) -> Any:
        return self.knobs.get(name, default)

    # ── files ──────────────────────────────────────────────────────────────
    def path(self, name: str) -> Path:
        """A file inside the work directory. Escaping it is refused."""
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ExecutionError(f"{name!r} is not a relative path inside the work directory")
        return self.workdir / candidate

    def _note_write(self, name: str) -> None:
        if not self.enforce_writes or not self._declared:
            return
        for declared in self._declared:
            if name == declared or name.startswith(declared.rstrip("/") + "/"):
                self.writes_made.append(name)
                return
        raise ExecutionError(
            f"node {self.node_id!r} wrote {name!r}, which it does not declare; "
            f"declared writes: {list(self._declared)}"
        )

    def write_text(self, name: str, text: str) -> Path:
        self._note_write(name)
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def append_text(self, name: str, text: str) -> Path:
        self._note_write(name)
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as handle:
            handle.write(text)
        return target

    def read_text(self, name: str) -> str:
        return self.path(name).read_text()

    def write_json(self, name: str, payload: Any) -> Path:
        import json

        return self.write_text(name, json.dumps(payload, indent=2, default=str))

    def read_json(self, name: str, default: Any = None) -> Any:
        import json

        target = self.path(name)
        if not target.exists():
            return default
        try:
            return json.loads(target.read_text())
        except json.JSONDecodeError:
            return default

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    # ── facts ──────────────────────────────────────────────────────────────
    def set_fact(self, name: str, value: Any) -> None:
        """Set a run fact. A node may only set facts it declares in ``produces``."""
        self.facts[name] = value
        self._produced.add(name)

    def _internal_fact(self, name: str, value: Any) -> None:
        """A fact the run itself records, not one a node claims to produce."""
        self.facts[name] = value

    def _check_produces(self) -> None:
        if not self.enforce_writes:
            return
        extra = sorted(self._produced - set(self._declared_facts))
        if extra:
            raise ExecutionError(
                f"node {self.node_id!r} produced the fact(s) {extra}, which it does "
                f"not declare; declared facts: {list(self._declared_facts)}"
            )


def render(template: str, ctx: RunContext) -> str:
    """Fill ``{{name}}`` from the run's knobs and facts.

    Deliberately not :meth:`str.format`: a prompt is prose full of braces, and a
    single stray one must not be able to make an agent node unrunnable.
    """
    import re

    values = {**ctx.knobs, **ctx.facts}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in values:
            return match.group(0)
        return str(values[key])

    return re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", replace, template)


#: The file a data node's current item is written to, for the pass that reads it.
CURRENT_ITEM = "current_item.json"

#: Where a run says which items a data node should read, when they are not the ones
#: the mode's own graph declares. A harness for a class of work is built once and fed
#: many times, so the source is run configuration rather than graph structure.
ITEMS_OVERRIDE = "items.json"


def load_items(node: Node, workdir: Path) -> list[dict[str, Any]]:
    """The items a data node iterates, in the order it declares them.

    One loader, so a mode means the same thing on either runtime: the item is handed
    to the pass as JSON rather than as a template variable, because a pass is a
    subgraph of nodes and only the first of them sees a prompt.

    The part the run directory names is applied last, by
    :func:`refactory_dsh.task.select_subset`; both runtimes load through this
    function, so both iterate the same part of the same items.
    """
    from refactory_dsh.task import select_subset

    return select_subset(full_items(node, workdir), workdir)


def full_items(node: Node, workdir: Path) -> list[dict[str, Any]]:
    """Every item a data node has, before the run directory's selection is applied.

    A node bound to a Task and naming no source takes its items from the Task, which
    is what makes ``task_ref`` a binding rather than a line in a prompt. Public
    because the outer loop sizes the partition it writes from the whole selection.
    """
    items = declared_items(node, workdir)
    if not items and node.task_ref:
        from refactory_dsh.task import load_task

        items = task_items(load_task(node.task_ref, workdir))
    return items


def declared_items(node: Node, workdir: Path) -> list[dict[str, Any]]:
    """The items the node's own source declares, before any Task or selection.

    A run directory may name a different source for this node, which is how a harness
    built for a class of work is pointed at items it has never seen: the graph stays
    what it was, and the run carries which items to read.
    """
    from refactory_dsh.items import source_override

    override = source_override(workdir, node.id)
    source_path = str(override.get("source_path") or node.source_path or "")
    source_format = str(override.get("source_format") or node.source_format or "directory")
    items: list[dict[str, Any]] = [] if override else [dict(item) for item in node.inline_items]
    if not items and source_path:
        source = workdir / source_path
        if not source.exists():
            source = Path(source_path)
        if not source.exists():
            raise ExecutionError(
                f"data node {node.id!r} names source_path {source_path!r}, "
                "which does not exist; bind the source before the run"
            )
        if source_format == "jsonl":
            for index, line in enumerate(source.read_text().splitlines()):
                if not line.strip():
                    continue
                payload = json.loads(line)
                items.append(payload if isinstance(payload, dict) else {"value": payload})
                items[-1].setdefault("id", str(index))
        elif source_format == "csv":
            import csv

            with source.open(newline="") as handle:
                items = [dict(row) for row in csv.DictReader(handle)]
        elif source_format == "directory":
            items = []
            for child in sorted(source.iterdir()):
                if not child.is_dir():
                    continue
                # An item is a directory, and it may describe itself in `item.json`:
                # those fields reach the pass the same way a Task's metadata does. The
                # directory's name stays the id, so a described item cannot rename
                # itself out of the selection it was listed in.
                item: dict[str, Any] = {"id": child.name, "path": str(child)}
                described = child / "item.json"
                if described.exists():
                    try:
                        loaded = json.loads(described.read_text())
                    except json.JSONDecodeError as exc:
                        raise ExecutionError(
                            f"item {child.name!r} of data node {node.id!r} carries an "
                            f"item.json that is not JSON: {exc}"
                        ) from exc
                    if isinstance(loaded, dict):
                        for key, value in loaded.items():
                            item.setdefault(key, value)
                items.append(item)
        else:
            raise ExecutionError(
                f"data node {node.id!r} declares source_format "
                f"{source_format!r}; known: directory, jsonl, csv"
            )
    limit = override.get("limit") or node.limit
    if limit:
        items = items[: int(limit)]
    return items


def task_items(task: "Task") -> list[dict[str, Any]]:
    """A Task's instances as the items a data node iterates."""
    return [
        {"id": instance.id, "path": instance.path, **instance.metadata}
        for instance in task.instances()
    ]


def _ask_model(model: Any, prompt: str, *, system: str, workdir: Path) -> str:
    """One completion, telling the client where the node's artifacts belong.

    ``workdir`` is not part of the vendored ``ModelClient`` protocol — the optimizer
    needs only text back — so a client that does not accept it is asked without it. The
    client that spawns an agent has to know: that agent has tools and writes files with
    relative paths, and a completion run from the checkout writes a node's artifacts
    into the repository. The reviewing nodes did exactly that, leaving the run's own
    `reviews/` empty while the QA synthesis read nothing.
    """
    import inspect

    kwargs: dict[str, Any] = {"system": system}
    try:
        parameters = inspect.signature(model.complete).parameters
    except (TypeError, ValueError):  # pragma: no cover - a builtin or a stub
        parameters = {}
    if "workdir" in parameters:
        kwargs["workdir"] = workdir
    return str(model.complete(prompt, **kwargs))


#: Where a run records how far it got, so a failure does not discard the work.
#:
#: Written after every node (the id of the next one) and when a node fails (the id of
#: the one that failed, with the error). An authoring run's artifacts already persist
#: in the run directory; this is what makes them usable by a restart instead of only
#: by a reader.
PROGRESS = ".refactory/progress.json"


@dataclass
class Executor:
    """Walk a graph from its start node to a terminal."""

    graph: Graph
    ctx: RunContext
    max_steps: int = 2000
    steps: list[Step] = field(default_factory=list)
    model_calls: int = 0
    _reserve_step: Callable[[], bool] | None = field(default=None, repr=False)

    _gate_iterations: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _outcome: str = field(default="pending", init=False, repr=False)
    _next_after_completion: str | None = field(default=None, init=False, repr=False)
    _background: list[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Fresh contexts used by resume still need the logical declarations of
        # gates that completed before the saved boundary.
        self.ctx._gate_nodes = {node.id: node for node in self.graph.nodes.values() if node.kind == "gate"}

    def _record(
        self, node: str, *, error: str | None = None,
        branches: dict[str, Any] | None = None,
        schedule: dict[str, Any] | None = None,
        outcome: str = "pending",
    ) -> None:
        """Write how far the run got, so a restart can continue instead of starting over.

        Best effort: a run directory that cannot be written to is the caller's
        problem, not a reason to fail a run that is otherwise going fine.
        """
        import json

        path = self.ctx.workdir / PROGRESS
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(".tmp")
            pending.write_text(json.dumps({
                "node": node,
                "error": error,
                "facts": self.ctx.facts,
                "knobs": self.ctx.knobs,
                "task": self.ctx.task,
                "steps": len(self.steps),
                "outcome": outcome,
                "graph": self.graph.to_state(),
                **({"model_context": self.ctx.model.context()} if callable(getattr(self.ctx.model, "context", None)) else {}),
                **({"branches": branches} if branches is not None else {}),
                **({"schedule": schedule} if schedule is not None else {}),
            }, indent=2, default=str))
            pending.replace(path)
        except OSError:  # pragma: no cover - an unwritable run directory
            pass

    def run(self, *, start: str | None = None) -> RunResult:
        """Finish owned advisory work before publishing the terminal result."""
        from refactory_dsh.owned_process import _termination_as_exit

        # Branch model/director calls run in threads and cannot install signal
        # handlers. The main owner must unwind through run_fork's cancellation
        # and join before exiting on its default SIGTERM path.
        with _termination_as_exit():
            return self._run_owned(start=start)

    def _run_owned(self, *, start: str | None = None) -> RunResult:
        prior_workers = set((self.ctx.workdir / ".refactory/workers").glob("*.json"))
        try:
            result = self._run(start=start)
        except BaseException:
            self._settle_background(cancel=True)
            raise
        original_error = result.error
        background_error = self._settle_background(cancel=not result.exit_ok)
        if background_error:
            result.error = f"{result.error}; {background_error}" if result.error else background_error
        if self.ctx.facts.get("create_fidelity") and self.ctx.facts.get("create_fidelity_precheck_ok") is False and result.error is None:
            result.error = str(self.ctx.facts.get("create_fidelity_precheck_error") or "Create fidelity final precheck failed")
        result.facts = dict(self.ctx.facts)
        result.exit_ok = result.error is None
        result.model_calls = self.model_calls
        if self.ctx.facts.get("create_fidelity"):
            result.model_calls = max(result.model_calls, len(set((self.ctx.workdir / ".refactory/workers").glob("*.json")) - prior_workers))
            if result.exit_ok:
                self._record("", outcome="completed")
            elif original_error is None or background_error is not None:
                # A foreground refusal already recorded its resumable node.
                # Only failures discovered while settling terminal work need a
                # new marker; an empty node would destroy approval/resume state.
                node = self.steps[-1].node if self.steps else (start or self.graph.start)
                self._record(node, error=result.error,
                             outcome="unknown" if "cleanup unconfirmed" in result.error else "failed")
        return result

    def _settle_background(self, *, cancel: bool = False) -> str | None:
        from concurrent.futures import TimeoutError

        errors = []
        for future, pool, child, stopped, deadline in self._background:
            if cancel:
                stopped.set()
            try:
                future.result(timeout=max(0.01, deadline - time.monotonic()))
                for name in child.ctx._produced:
                    self.ctx.facts[name] = child.ctx.facts[name]
            except TimeoutError:
                stopped.set()
                errors.append("advisory worker did not settle within its stage allowance; child cleanup unconfirmed")
            except Exception as exc:
                errors.append(f"advisory worker failed: {type(exc).__name__}: {exc}")
            finally:
                self.model_calls += child.model_calls
                pool.shutdown(wait=False, cancel_futures=True)
        self._background.clear()
        return "; ".join(errors) if errors else None

    def _run(self, *, start: str | None = None) -> RunResult:
        started = time.time()
        visits = self._gate_iterations
        current: str | None = start or self.graph.start or None
        error: str | None = None
        try:
            retries = loop_edges(self.graph)
        except ValueError as exc:
            error = str(exc)
            self._record(current or "", error=error)
            return RunResult(facts=dict(self.ctx.facts), exit_ok=False, error=error,
                             workdir=self.ctx.workdir)

        while current is not None:
            node = self.graph.nodes.get(current)
            if node is None:
                error = f"node {current!r} does not exist"
                break
            if len(self.steps) >= self.max_steps:
                error = f"the run exceeded {self.max_steps} steps"
                break
            if node.kind == "fork":
                from .fork import run_fork

                self.steps.append(Step(node.id, node.kind, 0.0))
                try:
                    error = run_fork(self, node)
                except Exception as exc:
                    error = f"{node.id}: {type(exc).__name__}: {exc}"
                    self._record(node.id, error=error)
                if error is not None:
                    break
                current = self.graph.out_edges(node.id)[0].target
                continue
            if node.kind == "join":
                self.steps.append(Step(node.id, node.kind, 0.0))
                out = self.graph.out_edges(node.id)
                current = out[0].target if out else None
                continue
            if node.kind == "data":
                # A data node runs its subgraph once per item, in order, and the item
                # is on disk for the ops and the agent nodes of that pass to read —
                # the same contract remote-factory's executor gives them. An empty
                # selection is a fact (`items_processed` is 0), not a crash and not a
                # silent success.
                self.steps.append(Step(node.id, node.kind, 0.0))
                try:
                    items = load_items(node, self.ctx.workdir)
                    # A Task bound to the node supplies the items when the node names
                    # none, and judges each pass. Loading it is what makes `task_ref` a
                    # binding rather than a line in a prompt.
                    from refactory_dsh.task import instance_of, load_task

                    task = load_task(node.task_ref, self.ctx.workdir) if node.task_ref else None
                except Exception as exc:
                    error = f"{node.id}: {type(exc).__name__}: {exc}"
                    break
                scores: list[dict[str, Any]] = []
                for item in items:
                    instance = instance_of(item)
                    if task is not None:
                        task.setup(instance, self.ctx.workdir)
                    # The item is written as the data node, so the declaration that
                    # governs it is the data node's — not the last node of the pass
                    # before it.
                    self.ctx.node_id = node.id
                    self.ctx._declared = node.writes  # noqa: SLF001
                    self.ctx._declared_facts = node.produces  # noqa: SLF001
                    self.ctx._produced = set()  # noqa: SLF001
                    self.ctx.write_json(CURRENT_ITEM, item)
                    branch_error = self._walk_branch(
                        node.subgraph_entry, node.subgraph_exit, include_barrier=True, iterations={}
                    )
                    if branch_error is not None:
                        error = branch_error
                        break
                    if task is not None:
                        verdict = task.verify(instance, self.ctx.workdir)
                        if verdict is not None:
                            scores.append({
                                "id": instance.id,
                                "passed": verdict.passed,
                                "score": verdict.score,
                            })
                if error is not None:
                    break
                if scores and "item_scores" in node.produces:
                    self.ctx.set_fact("item_scores", scores)
                out = self.graph.out_edges(node.id)
                current = out[0].target if out else None
                continue

            self.ctx.iteration = len(self.steps)
            self.ctx.node_id = current
            self.ctx._declared = node.writes  # noqa: SLF001 - the run owns the context
            self.ctx._declared_facts = node.produces  # noqa: SLF001
            self.ctx._produced = set()  # noqa: SLF001
            step_started = time.time()
            verdict: str | None = None
            try:
                current, verdict = self._visit(node)
                self.ctx._check_produces()  # noqa: SLF001
                self._count_retry(node, current, verdict, retries, visits)
            except Exception as exc:  # a failing node ends the run, visibly
                self.steps.append(
                    Step(node.id, node.kind, time.time() - step_started,
                         error=f"{type(exc).__name__}: {exc}")
                )
                error = f"{node.id}: {type(exc).__name__}: {exc}"
                from refactory_dsh.budgets import BudgetExhausted
                from concurrent.futures import CancelledError
                import subprocess

                if isinstance(exc, BudgetExhausted) and self._outcome == "completed":
                    self._record(self._next_after_completion or "", error=error, outcome="pending")
                else:
                    unknown = self._outcome == "running" and (
                        isinstance(exc, (BudgetExhausted, CancelledError, subprocess.TimeoutExpired))
                        or "timed out" in str(exc).lower()
                    )
                    self._record(node.id, error=error,
                                 outcome="unknown" if unknown else "failed")
                current = None
                break
            self.steps.append(
                Step(node.id, node.kind, time.time() - step_started, verdict=verdict)
            )
            if current is not None:
                self._record(current)

        if error is None:
            for step in self.steps:
                if step.error:
                    error = step.error
                    break
        if error is None and self.ctx.budget is not None:
            from refactory_dsh.budgets import BudgetExhausted

            try:
                self.ctx.budget.checkpoint()
            except BudgetExhausted as exc:
                error = str(exc)

        return RunResult(
            facts=dict(self.ctx.facts),
            iterations=len(self.steps),
            duration_s=round(time.time() - started, 4),
            exit_ok=error is None,
            error=error,
            workdir=self.ctx.workdir,
            model_calls=self.model_calls,
        )

    # ── one branch of a fork, or one pass of a data node ───────────────────
    def _walk_branch(
        self, start: str, barrier: str | None, *, include_barrier: bool = False,
        iterations: dict[str, int] | None = None,
    ) -> str | None:
        """Run from ``start`` up to ``barrier``, and report why it failed.

        A fork stops *before* its barrier, because the barrier is the node the walk
        continues at. A data node's pass includes its ``subgraph_exit``: that node is
        the last one of the subgraph, not the one after it.
        """
        current: str | None = start
        visits = self._gate_iterations if iterations is None else iterations
        try:
            retries = loop_edges(self.graph)
        except ValueError as exc:
            return str(exc)
        while current is not None and (current != barrier or include_barrier):
            if include_barrier and current == barrier:
                barrier = None  # the exit is visited once, then the pass ends
            node = self.graph.nodes.get(current)
            if node is None:
                return f"node {current!r} does not exist"
            if len(self.steps) >= self.max_steps or (self._reserve_step and not self._reserve_step()):
                return f"the run exceeded {self.max_steps} steps"
            self.ctx.iteration = len(self.steps)
            self.ctx.node_id = current
            self.ctx._declared = node.writes  # noqa: SLF001 - the run owns the context
            self.ctx._declared_facts = node.produces  # noqa: SLF001
            self.ctx._produced = set()  # noqa: SLF001
            started = time.time()
            verdict: str | None = None
            try:
                current, verdict = self._visit(node)
                self.ctx._check_produces()  # noqa: SLF001
                self._count_retry(node, current, verdict, retries, visits)
            except Exception as exc:  # a failing branch ends the run, visibly
                self.steps.append(
                    Step(node.id, node.kind, time.time() - started,
                         error=f"{type(exc).__name__}: {exc}")
                )
                return f"{node.id}: {type(exc).__name__}: {exc}"
            self.steps.append(
                Step(node.id, node.kind, time.time() - started, verdict=verdict)
            )
        return None

    @staticmethod
    def _count_retry(
        node: Node, target: str | None, verdict: str | None,
        retries: set[tuple[str, str, str]], visits: dict[str, int],
    ) -> None:
        if (node.id, target, verdict or "") not in retries:
            return
        count = visits.get(node.id, 0) + 1
        visits[node.id] = count
        if count > node.max_iterations:
            raise ExecutionError(
                f"gate {node.id!r} exceeded max_iterations of {node.max_iterations}; "
                f"retry {verdict!r} was refused before dispatching {target!r}"
            )

    # ── one node ───────────────────────────────────────────────────────────
    def _visit(self, node: Node) -> tuple[str | None, str | None]:
        """Run one node and return ``(next node id, verdict)``."""
        from refactory_dsh.live import emit

        if self.ctx.facts.get("create_fidelity") and not node.blocking and node.kind == "agent":
            import copy
            import threading
            from concurrent.futures import ThreadPoolExecutor
            from refactory_dsh.model import DEFAULT_TIMEOUT_S
            from refactory_dsh.owned_process import SHUTDOWN_GRACE_S

            stopped = threading.Event()
            model = self.ctx.model
            if callable(getattr(model, "with_cancellation", None)):
                model = model.with_cancellation(stopped)
            child_ctx = RunContext(self.ctx.workdir, knobs=copy.deepcopy(self.ctx.knobs),
                                   facts=copy.deepcopy(self.ctx.facts), model=model,
                                   task=copy.deepcopy(self.ctx.task), memory=self.ctx.memory,
                                   enforce_writes=self.ctx.enforce_writes)
            child_ctx.node_id = node.id
            child_ctx._declared, child_ctx._declared_facts = node.writes, node.produces
            child_ctx.budget = self.ctx.budget
            child = Executor(self.graph, child_ctx)
            child._reserve_step = lambda: True  # the owner alone writes progress
            def archive() -> None:
                child._visit(node.with_field("blocking", True))
                child_ctx._check_produces()
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="refactory-archive")
            future = pool.submit(archive)
            allowance = node.timeout_s or DEFAULT_TIMEOUT_S
            if callable(getattr(model, "effective_deadlines", None)):
                selected = model.for_node(node.id, role=node.role).with_node_timeout(node.timeout_s)
                allowance = selected.effective_deadlines()["stage_timeout_s"]
            self._background.append((future, pool, child, stopped,
                                     time.monotonic() + allowance + SHUTDOWN_GRACE_S + 2))
            out = self.graph.out_edges(node.id)
            return (out[0].target if out else None), None

        self._outcome = "pending"
        if self.ctx.budget is not None:
            self.ctx.budget.checkpoint_node(node.resource_phase)
        self._outcome = "running"
        if self._reserve_step is None:
            self._record(node.id, outcome="running")
        emit(self.ctx.workdir, "node_started", node_id=node.id, state="running", summary=node.notes)
        try:
            result = self._dispatch(node)
        except BaseException as exc:
            emit(self.ctx.workdir, "node_failed", node_id=node.id, state="failed", summary=f"{type(exc).__name__}: {exc}")
            raise
        self._outcome = "completed"
        self._next_after_completion = result[0]
        emit(self.ctx.workdir, "node_completed", node_id=node.id, state="completed", summary=node.notes)
        if self.ctx.budget is not None:
            self.ctx.budget.checkpoint()
        return result

    def _dispatch(self, node: Node) -> tuple[str | None, str | None]:
        """Execute a node after resource admission and the durable start marker."""
        if node.kind == "op":
            fn = resolve(node.call)
            previous = self.ctx.model
            if callable(getattr(previous, "with_node_timeout", None)):
                self.ctx.model = previous.with_node_timeout(node.timeout_s)
            try:
                produced = fn(self.ctx)
            finally:
                self.ctx.model = previous
            if isinstance(produced, dict):
                for key, value in produced.items():
                    self.ctx.set_fact(key, value)
        elif node.kind == "agent":
            self._visit_agent(node)
        else:
            return self._visit_gate(node)
        out = self.graph.out_edges(node.id)
        return (out[0].target if out else None), None

    def _visit_agent(self, node: Node) -> None:
        if self.ctx.model is None:
            raise ExecutionError(f"agent node {node.id!r} needs a model client")
        from refactory_dsh.agent_delivery import begin, finish

        client = self.ctx.model
        if callable(getattr(client, "for_node", None)):
            client = client.for_node(node.id, role=node.role)
        if callable(getattr(client, "with_node_timeout", None)):
            client = client.with_node_timeout(node.timeout_s)
        route = client.route() if callable(getattr(client, "route", None)) else {}
        if self.ctx.facts.get("create_fidelity") and node.role != "builder" and route.get("worker_backend", "dsh") != "dsh":
            raise ExecutionError(f"Create fidelity role {node.role!r} requires DSH execution")
        prompt = render(node.prompt_template, self.ctx)
        if self.ctx.facts.get("create_fidelity"):
            from refactory_dsh.create_fidelity_ops import agent_prompt
            prompt = agent_prompt(self.ctx, node)
        delivery = begin(self.ctx.workdir, node.id, prompt, list(node.writes),
                         reads=list(node.reads),
                         model_context=client.context() if callable(getattr(client, "context", None)) else None)
        self.model_calls += 1
        if self.ctx.budget is not None and not callable(getattr(client, "with_budget", None)):
            self.ctx.budget.require_usage_capability(False)
        output = _ask_model(
            client, delivery["prompt"], system=node.notes, workdir=self.ctx.workdir
        )
        finish(self.ctx.workdir, node.id, final_response=output,
               capture_final=route.get("worker_backend", "dsh") == "dsh")
        if node.produces:
            # Text-producing nodes expose the delivered artifact, which can have
            # been written by tools while the final response only summarizes it.
            value = output
            if node.writes:
                try:
                    value = self.ctx.read_text(node.writes[0])
                except (UnicodeDecodeError, IsADirectoryError):
                    # Binary/directory delivery has no textual fact to substitute.
                    pass
            self.ctx.set_fact(node.produces[0], value)

    def _answered_verdict(self, node: Node) -> str:
        """The verdict for a gate a person or an agent answers.

        A headless walk has nobody to ask, so it **refuses** rather than inventing an
        answer. That is the whole point of :attr:`Node.gate_kind`: a run that pauses
        for a person cannot quietly decide for them, and a run that did would be a
        result nobody can account for. The answer is supplied through
        ``RunContext.verdicts`` for explicit upfront authorization, or consumed from
        the intent ledger for the current review inputs.
        """
        verdict = (self.ctx.verdicts or {}).get(node.id, "")
        if not verdict and node.gate_kind == "user":
            from refactory_dsh.steering import Ledger

            ledger = Ledger(self.ctx.workdir)
            labels = [str(edge.condition) for edge in self.graph.out_edges(node.id)]
            verdict = ledger.take_verdict(node.id, ledger.visit(node.id, node.reads), labels)
        if not verdict:
            raise ExecutionError(
                f"gate {node.id!r} is answered by {node.gate_kind!r} and this run has "
                "no verdict for it. A headless run cannot answer a gate meant for a "
                "person or for an agent; pass one in RunContext.verdicts (an "
                "interactive run collects a fresh answer) or make the "
                "gate `fn`."
            )
        return str(verdict)

    def _visit_gate(self, node: Node) -> tuple[str, str]:
        if self.ctx.facts.get("create_fidelity") and node.gate_kind in ("agent", "user"):
            from refactory_dsh.create_fidelity_gates import evaluate
            self.ctx._gate_labels = [str(edge.condition) for edge in self.graph.out_edges(node.id)]
            if not hasattr(self.ctx, "_gate_nodes"):
                self.ctx._gate_nodes = {}
            self.ctx._gate_nodes[node.id] = node
            if node.gate_kind == "agent":
                self.model_calls += 1
            verdict = evaluate(self.ctx, node)
        elif node.gate_kind != "fn":
            verdict = self._answered_verdict(node)
        else:
            fn: Callable[[RunContext], Any] = resolve(node.evaluator)
            verdict = fn(self.ctx)
        if not isinstance(verdict, str) or not verdict:
            raise ExecutionError(
                f"gate {node.id!r} returned {verdict!r}; a gate must return a verdict label"
            )
        out = self.graph.out_edges(node.id, condition=verdict)
        if not out:
            labels = sorted(e.condition or "" for e in self.graph.out_edges(node.id))
            raise ExecutionError(
                f"gate {node.id!r} returned {verdict!r}, which is not one of its "
                f"verdicts {labels}"
            )
        self.ctx._internal_fact(f"{node.id}_verdict", verdict)  # noqa: SLF001
        return out[0].target, verdict


def execute(
    graph: Graph,
    workdir: Path,
    *,
    knobs: dict[str, Any] | None = None,
    facts: dict[str, Any] | None = None,
    verdicts: dict[str, str] | None = None,
    model: ModelClient | None = None,
    task: dict[str, Any] | None = None,
    memory: Any = None,
    enforce_writes: bool = True,
    max_steps: int = 2000,
    start: str | None = None,
    resource_limits: Any = None,
    shared_budget: Any = None,
) -> RunResult:
    """Build a context and run the graph once.

    Two things come from the run directory before anything executes, because a mode
    author cannot be the thing that decides whether they happen:

    - The **intent ledger** is folded into every agent node, so a run a person
      steered carries what they said. Inert when there is no ledger.
    - A user gate consumes a pending answer for its current review inputs once.
      An explicit ``verdicts`` argument is standing upfront authorization.

    ``start`` and ``facts`` are how a failed run continues: the node it died on, and
    the facts the nodes before it produced. Both come from the run's own
    ``.refactory/progress.json``; a caller that passes neither runs the graph from
    its start with the facts it was given.
    """
    from refactory_dsh.steering import for_run

    workdir = Path(workdir)
    from refactory_dsh.budgets import LEDGER, ResourceLimits, RunBudget

    budget = shared_budget or RunBudget.inherited(model)
    if budget is None and start is None and (workdir / LEDGER).exists():
        raise ValueError("this directory already has a resource ledger; use a new run directory or explicit resume")
    own_budget = budget is None and resource_limits is not None
    if own_budget:
        limits = resource_limits if isinstance(resource_limits, ResourceLimits) else ResourceLimits.from_dict(resource_limits)
        budget = RunBudget.open(workdir, limits) if start is not None else RunBudget.create(workdir, limits)
        budget.resume()
    if budget is not None:
        model = budget.bind_model(model)
    graph = for_run(graph, workdir)
    ctx = RunContext(
        workdir,
        knobs=knobs,
        facts=facts,
        verdicts=verdicts,
        model=model,
        task=task,
        memory=memory,
        enforce_writes=enforce_writes,
    )
    ctx.budget = budget
    try:
        result = Executor(graph, ctx, max_steps=max_steps).run(start=start)
        if budget is not None:
            result.facts["resources"] = budget.snapshot()
        return result
    finally:
        if own_budget and budget is not None:
            budget.pause()
