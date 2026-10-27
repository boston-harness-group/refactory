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

import copy
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback as traceback_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .graph import Edge, Graph, Node
from .loops import loop_edges
from .mode import RunResult
from .record import RunRecord
from .resolve import resolve

if TYPE_CHECKING:
    from refactory_dsh.task import Task

# ── forkserver context (module-level, shared across all Executor instances) ──
_forkserver_ctx = multiprocessing.get_context("forkserver")
_forkserver_modules: set[str] = set()


def _ensure_forkserver_preload(graph: Graph) -> None:
    """Collect modules from the graph's ops and fn gates and preload them.

    The forkserver is a separate single-threaded process; forking from it is
    safe while the walker runs fork branches on threads.  It has the same
    picklable-arguments contract as spawn.

    ``set_forkserver_preload`` is idempotent before the server starts and is a
    hint afterwards (the server restarts with the updated list if it dies).
    """
    global _forkserver_modules
    modules: set[str] = {"refactory_dsh.nn.executor", "refactory_dsh.run"}
    for node in graph.nodes.values():
        for target in (node.call, node.evaluator):
            if not target:
                continue
            if ":" in target:
                mod_name = target.partition(":")[0]
            else:
                mod_name = target.rpartition(".")[0]
            if mod_name:
                modules.add(mod_name)
    if not modules.issubset(_forkserver_modules):
        _forkserver_modules = _forkserver_modules | modules
        _forkserver_ctx.set_forkserver_preload(sorted(_forkserver_modules))


class ExecutionError(RuntimeError):
    """A run could not be carried out."""


class OpTimeoutError(RuntimeError):
    """An op or fn gate exceeded its per-node timeout."""

    def __init__(self, node_id: str, timeout_s: float):
        self.node_id = node_id
        self.timeout_s = timeout_s
        super().__init__(f"op '{node_id}' timed out after {timeout_s:.1f}s")


class GateStop(Exception):
    """The walker stopped at a user gate that needs a person's answer."""

    def __init__(self, gate: str, reads: list[str], edges: list[Edge]):
        self.gate = gate
        self.reads = reads
        self.edges = edges
        super().__init__(f"gate {gate!r} is waiting for a person")


class HaltStop(Exception):
    """A halt verdict ended the run."""

    def __init__(self, gate: str, reason: str = "verdict"):
        self.gate = gate
        self.reason = reason
        super().__init__(f"gate {gate!r} halted the run")


def route_verdict(verdict: str, edges: list[Edge], gate_kind: str) -> tuple[Edge | None, str]:
    """Route a verdict to an edge, matching the spine's ``Spine.verdict()``.

    Returns ``(edge, reason)``. When the edge is ``None``, the run is halted
    with the given reason (``"verdict"``).

    - Match labels case-insensitively.
    - ``proceed`` with no proceed edge falls through to the unconditional edge.
    - ``halt`` follows a halt edge if there is one; otherwise returns
      ``(None, "verdict")`` — terminal halt.
    - An unknown verdict raises :class:`ExecutionError`.
    """
    lower = verdict.lower()
    # Direct match (case-insensitive)
    for edge in edges:
        if (edge.condition or "").lower() == lower:
            return edge, ""
    # proceed falls through to unconditional
    if lower == "proceed":
        for edge in edges:
            if not edge.condition:
                return edge, ""
    # halt with no halt edge is terminal
    if lower == "halt":
        return None, "verdict"
    raise ExecutionError(
        f"verdict {verdict!r} is not one of {sorted(e.condition or '' for e in edges)}"
    )


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
    #: Why a gate gave its verdict, when it said: the same reason the spine reports
    #: under ``ran``, so a headless run explains a reloop as a driven one does.
    reason: str | None = None


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
        #: headless run, which is why such a gate stops rather than deciding.
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
        self._declared_reads: tuple[str, ...] = ()
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

    @property
    def declared_reads(self) -> tuple[str, ...]:
        """The current node's declared reads, for gate discovery."""
        return self._declared_reads

    def _check_produces(self) -> None:
        if not self.enforce_writes:
            return
        extra = sorted(self._produced - set(self._declared_facts))
        if extra:
            raise ExecutionError(
                f"node {self.node_id!r} produced the fact(s) {extra}, which it does "
                f"not declare; declared facts: {list(self._declared_facts)}"
            )


class Verdict(str):
    """A gate's verdict label, carrying why the gate gave it.

    It *is* the label, so routing, comparisons and every caller that reads a
    gate's return as a string are unchanged. The node runner prints ``reason``
    before the label, which is how the spine hands a reloop's cause to the node
    the run goes back to and to the CEO driving it. A reason that opens with its
    own label (``"reloop: …"``, the form gate facts are written in) drops it, so
    a reader is not told the verdict twice.
    """

    reason: str

    def __new__(cls, label: str, reason: str = "") -> Verdict:
        verdict = super().__new__(cls, label)
        verdict.reason = reason.removeprefix(f"{label}: ")
        return verdict


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


def _op_child_target(conn: Any, args_dict: dict[str, Any]) -> None:
    """Target function for the spawned child process running an op or fn gate."""
    os.setsid()

    def _sigterm_handler(signum: int, frame: Any) -> None:
        raise SystemExit(f"terminated by signal {signum}")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Restore parent's sys.path so resolve() finds the same modules.
    sys.path[:] = args_dict["sys_path"]

    try:
        from refactory_dsh.run import build_op_context

        model: Any = None
        model_info = args_dict.get("model_info")
        if model_info:
            kind = model_info.get("kind")
            if kind == "dsh":
                from refactory_dsh.model import DshModelClient
                model = DshModelClient(**model_info["context"])
            elif kind is not None:
                # Stand-in for non-DSH clients: not None so prepare/create_ops
                # skip readiness, and every method raises so nothing silently proceeds.
                class _StandIn:
                    def __getattr__(self, name: str) -> Any:
                        raise RuntimeError(f"model.{name}() called in child process")
                    def complete(self, *a: Any, **kw: Any) -> str:
                        raise RuntimeError("model.complete() called in child process")
                model = _StandIn()

        ctx = build_op_context(
            Path(args_dict["workdir"]),
            args_dict["node_id"],
            knobs=args_dict.get("knobs"),
            facts=args_dict.get("facts"),
            task=args_dict.get("task"),
        )
        # Override declarations with the node's own (not from disk).
        ctx._declared = tuple(args_dict.get("writes", ()))
        ctx._declared_reads = tuple(args_dict.get("reads", ()))
        ctx._declared_facts = tuple(args_dict.get("produces", ()))

        # Set model AFTER context creation so RunBudget.inherited sees None,
        # not the stand-in.  The stand-in is only for ops that check ctx.model.
        model_info_2 = args_dict.get("model_info")
        if model_info_2 and model_info_2.get("kind") == "other":
            ctx.model = model
        elif model is not None:
            ctx.model = model

        fn = resolve(args_dict["callable_target"])
        result = fn(ctx)

        conn.send(("ok", {
            "result": result,
            "facts": dict(ctx.facts),
            "writes_made": list(ctx.writes_made),
            "_produced": list(ctx._produced),
        }))
    except BaseException as exc:
        conn.send(("error", {
            "type": type(exc).__qualname__,
            "module": type(exc).__module__,
            "message": str(exc),
            "traceback": "".join(traceback_mod.format_exception(exc)),
        }))
    finally:
        conn.close()


@dataclass
class Executor:
    """Walk a graph from its start node to a terminal."""

    graph: Graph
    ctx: RunContext
    max_steps: int = 2000
    enable_recovery: bool = False
    steps: list[Step] = field(default_factory=list)
    model_calls: int = 0
    #: The completed nodes, gate verdicts and reloop feedback, kept by the spine's rules.
    record: RunRecord = field(default_factory=RunRecord)
    _reserve_step: Callable[[], bool] | None = field(default=None, repr=False)

    _gate_iterations: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _outcome: str = field(default="pending", init=False, repr=False)
    _next_after_completion: str | None = field(default=None, init=False, repr=False)
    _retries: set[tuple[str, str, str]] | None = field(default=None, init=False, repr=False)
    _status: str = field(default="", init=False, repr=False)
    #: The retry counters of the walk visiting the current node (a data pass has its own).
    _visits: dict[str, int] | None = field(default=None, init=False, repr=False)

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
                "file_fingerprints": self._file_fingerprints(),
                **({"model_context": self.ctx.model.context()} if callable(getattr(self.ctx.model, "context", None)) else {}),
                **({"branches": branches} if branches is not None else {}),
                **({"schedule": schedule} if schedule is not None else {}),
                **self.record.to_progress(),
            }, indent=2, default=str))
            pending.replace(path)
        except OSError:  # pragma: no cover - an unwritable run directory
            pass
        if self._reserve_step is not None:
            return  # a fork's branch: the run it belongs to writes the run's files
        running = [target for target, state in (branches or {}).items()
                   if state.get("status") in ("queued", "running")]
        nodes = running or ([node] if node else [])
        if outcome in ("refused", "halted"):
            self._write_status(outcome, [] if outcome == "halted" else nodes, error=error)
        elif error is not None:
            self._write_status("failed", nodes, error=error, failed=node or None)
        else:
            self._write_status("executing", nodes)

    def _file_fingerprints(self) -> dict[str, Any]:
        """Every declared input and output, fingerprinted as the spine fingerprints them in
        its checkpoint at each boundary it records (`recordStatus`), with the run's
        configuration: a resume refuses a run whose files changed since (dm-160)."""
        from refactory_dsh.recovery import fingerprint

        names = {name for node in self.graph.nodes.values() for name in (*node.reads, *node.writes)}
        if (self.ctx.workdir / ".refactory/run-config.json").exists():
            names.add(".refactory/run-config.json")
        return {name: fingerprint(self.ctx.workdir / name) for name in sorted(names)}

    def _write_status(self, status: str, nodes: list[str], *, error: str | None = None,
                      failed: str | None = None) -> None:
        """Write the run's facts and its status where the spine writes them (`facts.json`,
        `.refactory/spine-status.json`), so a walked run reads as a driven one does.
        Best effort, as the progress record is."""
        from refactory_dsh.run import write_facts

        try:
            write_facts(self.ctx.workdir, self.ctx.facts)
            self.record.write_status(self.ctx.workdir, self.graph, status, nodes, error=error, failed=failed)
            self._status = status
        except OSError:  # pragma: no cover - an unwritable run directory
            pass

    def _is_retry(self, gate: str, target: str, condition: str) -> bool:
        if self._retries is None:
            self._retries = loop_edges(self.graph)
        return (gate, target, condition) in self._retries

    def run(self, *, start: str | None = None) -> RunResult:
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
                self.record.complete(node)
                try:
                    error = run_fork(self, node)
                except GateStop as exc:
                    error = str(exc)
                    self._record(node.id, error=error, outcome="refused")
                except HaltStop as exc:
                    error = str(exc)
                    self._record(node.id, error=error, outcome="halted")
                except Exception as exc:
                    error = f"{node.id}: {type(exc).__name__}: {exc}"
                    self._record(node.id, error=error)
                if error is not None:
                    break
                current = self.graph.out_edges(node.id)[0].target
                continue
            if node.kind == "join":
                self.steps.append(Step(node.id, node.kind, 0.0))
                self.record.complete(node)
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
                    self.ctx._declared_reads = node.reads  # noqa: SLF001
                    self.ctx._declared_facts = node.produces  # noqa: SLF001
                    self.ctx._produced = set()  # noqa: SLF001
                    self.ctx.write_json(CURRENT_ITEM, item)
                    try:
                        branch_error = self._walk_branch(
                            node.subgraph_entry, node.subgraph_exit, include_barrier=True, iterations={}
                        )
                    except GateStop as exc:
                        branch_error = str(exc)
                        self._record(node.id, error=branch_error, outcome="refused")
                    except HaltStop as exc:
                        branch_error = str(exc)
                        self._record(node.id, error=branch_error, outcome="halted")
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
                self.record.complete(node)
                out = self.graph.out_edges(node.id)
                current = out[0].target if out else None
                continue

            self.ctx.iteration = len(self.steps)
            self.ctx.node_id = current
            self.ctx._declared = node.writes  # noqa: SLF001 - the run owns the context
            self.ctx._declared_reads = node.reads  # noqa: SLF001
            self.ctx._declared_facts = node.produces  # noqa: SLF001
            self.ctx._produced = set()  # noqa: SLF001
            step_started = time.time()
            verdict: str | None = None
            try:
                self._visits = visits
                current, verdict = self._try_visit(node, step_started)
                self.ctx._check_produces()  # noqa: SLF001
                self._count_retry(node, current, verdict, retries, visits)
            except GateStop as exc:
                error = str(exc)
                self._record(exc.gate, error=error, outcome="refused")
                current = None
                break
            except HaltStop as exc:
                error = str(exc)
                self._record(exc.gate, error=error, outcome="halted")
                current = None
                break
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
                elif isinstance(exc, OpTimeoutError):
                    self._record(node.id, error=error, outcome="failed")
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
                Step(node.id, node.kind, time.time() - step_started, verdict=verdict,
                     reason=getattr(verdict, "reason", None) or None)
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

        if self._reserve_step is None:
            if error is None:
                # The walk's end is a boundary of its own: a resume of it goes on from
                # nothing, as the spine's resume of a done run does.
                self._record("", outcome="completed")
                self._write_status("done", [])
            elif self._status not in ("refused", "halted", "failed"):
                self._write_status("failed", [current] if current else [], error=error, failed=current)

        gate_stop_id = None
        halted = False
        if isinstance(error, str) and "is waiting for a person" in error:
            # Extract gate id from the GateStop message pattern.
            import re as _re
            m = _re.search(r"gate '([^']+)' is waiting", error)
            if m:
                gate_stop_id = m.group(1)
        if isinstance(error, str) and "halted the run" in error:
            halted = True
        return RunResult(
            facts=dict(self.ctx.facts),
            iterations=len(self.steps),
            duration_s=round(time.time() - started, 4),
            exit_ok=error is None,
            error=error,
            workdir=self.ctx.workdir,
            model_calls=self.model_calls,
            gate_stop=gate_stop_id,
            halted=halted,
            gate_verdicts=dict(self.record.gate_verdicts),
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
            self.ctx._declared_reads = node.reads  # noqa: SLF001
            self.ctx._declared_facts = node.produces  # noqa: SLF001
            self.ctx._produced = set()  # noqa: SLF001
            started = time.time()
            verdict: str | None = None
            try:
                self._visits = visits
                current, verdict = self._try_visit(node, started)
                self.ctx._check_produces()  # noqa: SLF001
                self._count_retry(node, current, verdict, retries, visits)
            except (GateStop, HaltStop):
                raise
            except Exception as exc:  # a failing branch ends the run, visibly
                self.steps.append(
                    Step(node.id, node.kind, time.time() - started,
                         error=f"{type(exc).__name__}: {exc}")
                )
                return f"{node.id}: {type(exc).__name__}: {exc}"
            self.steps.append(
                Step(node.id, node.kind, time.time() - started, verdict=verdict,
                     reason=getattr(verdict, "reason", None) or None)
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

    # ── retry wrapper ──────────────────────────────────────────────────────
    def _try_visit(self, node: Node, step_started: float) -> tuple[str | None, str | None]:
        """Wrap ``_visit`` with FnNode-only one-retry under ``enable_recovery``."""
        if not self.enable_recovery or node.kind != "op":
            return self._visit(node)

        from refactory_dsh.budgets import BudgetExhausted

        facts_snapshot = copy.deepcopy(self.ctx.facts)
        writes_mark = len(self.ctx.writes_made)

        try:
            return self._visit(node)
        except BudgetExhausted:
            raise
        except OpTimeoutError:
            raise
        except Exception:
            from refactory_dsh.run import BuildLoopStop
            if isinstance(sys.exc_info()[1], BuildLoopStop):
                raise
            # Restore state to pre-attempt snapshot.
            self.ctx.facts = facts_snapshot
            self.ctx._produced = set()  # noqa: SLF001
            self.ctx.writes_made = self.ctx.writes_made[:writes_mark]
            self._outcome = "pending"
            self._next_after_completion = None

            # Retry once.
            result = self._visit(node)

            # Record the successful retry.
            self.steps.append(
                Step(node.id, node.kind, time.time() - step_started,
                     verdict="retried", error=None)
            )
            return result

    # ── effective timeout ──────────────────────────────────────────────────
    def _effective_timeout(self, node: Node) -> float | None:
        """Derive per-op timeout from the shared formula and clamp to budget."""
        if not self.enable_recovery:
            return None
        if node.kind == "op" or (node.kind == "gate" and node.gate_kind == "fn"):
            from refactory_dsh.lowering import command_timeout_s as _cmd_timeout_s

            state = {"knobs": {n: {"value": v} for n, v in self.ctx.knobs.items()}}
            raw = _cmd_timeout_s(node, state)
            if raw is None:
                # Unclassified ops stay in-process, with the run's budget: an op that
                # records usage through ctx.budget needs it, and a child gets none.
                # Giving them DSH's 3600 s default needs budget forwarding (dm-173).
                return None
            remaining = self.ctx.budget.remaining_s() if self.ctx.budget is not None else None
            if remaining is not None:
                raw = min(raw, remaining)
            return max(1.0, float(raw))
        return None

    # ── child-process-based op timeout ─────────────────────────────────────
    def _run_op_timed(self, callable_target: str, node: Node, timeout: float) -> Any:
        """Run a callable in a forkserver child process, killed at *timeout*."""
        _ensure_forkserver_preload(self.graph)
        parent_conn, child_conn = _forkserver_ctx.Pipe()

        model_info: dict[str, Any] | None = None
        if self.ctx.model is not None:
            from refactory_dsh.model import DshModelClient

            if isinstance(self.ctx.model, DshModelClient):
                model_info = {"kind": "dsh", "context": self.ctx.model.context()}
            else:
                model_info = {"kind": "other"}

        args_dict: dict[str, Any] = {
            "sys_path": list(sys.path),
            "workdir": str(self.ctx.workdir),
            "facts": dict(self.ctx.facts),
            "knobs": dict(self.ctx.knobs),
            "task": dict(self.ctx.task),
            "callable_target": callable_target,
            "node_id": node.id,
            "writes": tuple(node.writes),
            "reads": tuple(node.reads),
            "produces": tuple(node.produces),
            "model_info": model_info,
        }

        child = _forkserver_ctx.Process(target=_op_child_target, args=(child_conn, args_dict))
        child.start()
        child_conn.close()

        if parent_conn.poll(timeout):
            status, payload = parent_conn.recv()
            parent_conn.close()
            child.join(timeout=5)
        else:
            parent_conn.close()
            # Kill escalation: SIGTERM → grace → SIGKILL, matching DSH.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                try:
                    child.kill()
                except OSError:
                    pass
            child.join(timeout=5)
            if child.is_alive():
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        child.kill()
                    except OSError:
                        pass
                child.join(timeout=5)
            raise OpTimeoutError(node.id, timeout)

        if status == "ok":
            # Merge child state back into the parent's context.
            child_facts = payload.get("facts", {})
            for key, value in child_facts.items():
                if key not in self.ctx.facts or self.ctx.facts[key] != value:
                    self.ctx.facts[key] = value
            self.ctx._produced.update(payload.get("_produced", []))  # noqa: SLF001
            self.ctx.writes_made.extend(payload.get("writes_made", []))
            return payload.get("result")

        # Exception from child: reconstruct and re-raise.
        exc_type_name = payload.get("type", "RuntimeError")
        exc_msg = payload.get("message", "")
        exc_tb = payload.get("traceback", "")
        exc_module = payload.get("module", "builtins")

        reconstructed: BaseException | None = None
        try:
            import importlib as _il

            mod = _il.import_module(exc_module)
            cls = getattr(mod, exc_type_name, None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, BaseException):
                reconstructed = cls(exc_msg)
        except Exception:
            pass

        if reconstructed is not None:
            raise reconstructed
        raise RuntimeError(f"{exc_type_name}: {exc_msg}\n{exc_tb}")

    # ── one node ───────────────────────────────────────────────────────────
    def _visit(self, node: Node) -> tuple[str | None, str | None]:
        """Run one node and return ``(next node id, verdict)``."""
        from refactory_dsh.live import emit

        self._outcome = "pending"
        if self.ctx.budget is not None:
            self.ctx.budget.checkpoint_node(node.resource_phase)
        # A user gate with no answer yet waits for a person, which is the run's status, not
        # a visit: the spine logs no visit for it either. So its answer is read before the
        # visit starts, and a `GateStop` leaves no started or failed node behind.
        answer = self._answered_verdict(node) if node.kind == "gate" and node.gate_kind == "user" else None
        self._outcome = "running"
        if self._reserve_step is None:
            self._record(node.id, outcome="running")
        emit(self.ctx.workdir, "node_started", node_id=node.id, state="running", summary=render(node.notes, self.ctx))
        try:
            result = self._dispatch(node, answer)
        except BaseException as exc:
            emit(self.ctx.workdir, "node_failed", node_id=node.id, state="failed", summary=f"{type(exc).__name__}: {exc}")
            raise
        self._outcome = "completed"
        self._next_after_completion = result[0]
        if node.kind != "gate":  # a gate is recorded with its verdict
            self.record.complete(node)
        emit(self.ctx.workdir, "node_completed", node_id=node.id, state="completed", summary=render(node.notes, self.ctx))
        if self.ctx.budget is not None:
            self.ctx.budget.checkpoint()
        return result

    def _dispatch(self, node: Node, answer: str | None = None) -> tuple[str | None, str | None]:
        """Execute a node after resource admission and the durable start marker. ``answer``
        is a user gate's, read before its visit started."""
        if node.kind == "op":
            timeout = self._effective_timeout(node)
            if timeout is not None:
                produced = self._run_op_timed(node.call, node, timeout)
            else:
                fn = resolve(node.call)
                produced = fn(self.ctx)
            if isinstance(produced, dict):
                for key, value in produced.items():
                    self.ctx.set_fact(key, value)
        elif node.kind == "agent":
            self._visit_agent(node)
        else:
            return self._visit_gate(node, answer)
        out = self.graph.out_edges(node.id)
        return (out[0].target if out else None), None

    def _visit_agent(self, node: Node) -> None:
        if self.ctx.model is None:
            raise ExecutionError(f"agent node {node.id!r} needs a model client")
        from refactory_dsh.agent_delivery import fill_feedback, with_feedback

        client = self.ctx.model
        if callable(getattr(client, "for_node", None)):
            client = client.for_node(node.id, role=node.role)
        # The node's own bound on its session, as the spine applies it (#307).
        if node.timeout_s is not None and callable(getattr(client, "with_wall", None)):
            client = client.with_wall(node.timeout_s)
        # Re-fill feedback keys before rendering — a failure to re-derive
        # a declared key refuses the dispatch (dm-120).
        values = {**self.ctx.knobs, **self.ctx.facts}
        values, refilled_keys = fill_feedback(
            self.ctx.workdir, node.id, values,
            feedback_keys=node.feedback_keys if node.feedback_keys else None,
        )
        if refilled_keys:
            for k in refilled_keys:
                self.ctx._internal_fact(k, values[k])
        model_context = client.context() if callable(getattr(client, "context", None)) else None
        route = client.route() if callable(getattr(client, "route", None)) else {}
        from refactory_dsh import workspace as workspaces
        from refactory_dsh.node_retry import deliver

        def dispatch_workspace(retrying: bool):
            return workspaces.dispatched(
                self.ctx.workdir, node.id, node.workspace, knobs=self.ctx.knobs, facts=self.ctx.facts,
                route=route, budget=self.ctx.budget, fresh=retrying,
                runtime_root=Path(model_context["runtime_root"]) if (model_context or {}).get("runtime_root")
                else None)

        prompt = with_feedback(render(node.prompt_template, self.ctx), self.record.feedback.get(node.id, ""))
        if self.ctx.budget is not None and not callable(getattr(client, "with_budget", None)):
            self.ctx.budget.require_usage_capability(False)

        def ask(attempt: Any, sent: str, place: Any) -> str:
            self.model_calls += 1
            if place is None:
                return _ask_model(attempt, sent, system="", workdir=self.ctx.workdir)
            return workspaces.ask(place, attempt, sent, workdir=self.ctx.workdir, system="")

        output, _receipt = deliver(
            self.ctx.workdir, node.id, prompt=prompt, writes=list(node.writes),
            reads=list(node.reads), client=client, ask=ask,
            capture_final=route.get("worker_backend", "dsh") == "dsh", rule=node.retry,
            timeout_s=node.timeout_s, budget=self.ctx.budget, workspace=dispatch_workspace)
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

        A headless walk has nobody to ask, so it **stops** rather than inventing an
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
            if node.gate_kind == "user":
                raise GateStop(
                    node.id,
                    list(node.reads),
                    self.graph.out_edges(node.id),
                )
            raise ExecutionError(
                f"gate {node.id!r} is answered by {node.gate_kind!r} and this run has "
                "no verdict for it. A headless run cannot answer a gate meant for a "
                "person or for an agent; pass one in RunContext.verdicts (an "
                "interactive run collects a fresh answer) or make the "
                "gate `fn`."
            )
        return str(verdict)

    def _visit_gate(self, node: Node, answer: str | None = None) -> tuple[str, str]:
        if answer is not None:
            verdict = answer
        elif node.gate_kind != "fn":
            verdict = self._answered_verdict(node)
        else:
            timeout = self._effective_timeout(node)
            if timeout is not None:
                verdict = self._run_op_timed(node.evaluator, node, timeout)
            else:
                fn: Callable[[RunContext], Any] = resolve(node.evaluator)
                verdict = fn(self.ctx)
        if not isinstance(verdict, str) or not verdict:
            raise ExecutionError(
                f"gate {node.id!r} returned {verdict!r}; a gate must return a verdict label"
            )
        edges = self.graph.out_edges(node.id)
        selected, reason = route_verdict(verdict, edges, node.gate_kind)
        # The verdict is the run's record, kept where the spine keeps its own (its
        # checkpoint's `gate_verdicts`, here the progress record), never a fact: no node
        # reads it, and the spine's facts have none.
        if selected is None:
            # Terminal halt
            self.record.gate(self.graph, node.id, verdict, target=None, retry=False)
            raise HaltStop(node.id, reason=reason)
        why = str(getattr(verdict, "reason", "") or "")
        from refactory_dsh.run import _clipped

        retry = self._is_retry(node.id, selected.target, selected.condition or "")
        visits = self._gate_iterations if self._visits is None else self._visits
        if retry and visits.get(node.id, 0) + 1 > node.max_iterations:
            # Out of reloops: the spine records the gate halted and goes back over nothing
            # (`Spine.verdict`); the retry itself is refused below (`_count_retry`).
            self.record.gate(self.graph, node.id, "halt", target=None, retry=False)
        else:
            self.record.gate(self.graph, node.id, verdict, target=selected.target, retry=retry,
                             reason=_clipped(why.strip()))
        return selected.target, Verdict(selected.condition or "", reason=why)


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
    enable_recovery: bool = False,
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
    # A run that goes on from where it stopped goes on with its record: the nodes it
    # completed, its gates' verdicts and any feedback a reloop left.
    record = RunRecord()
    if start is not None:
        try:
            record = RunRecord.from_progress(json.loads((workdir / PROGRESS).read_text()))
        except (OSError, ValueError, AttributeError):
            pass
    try:
        result = Executor(graph, ctx, max_steps=max_steps, enable_recovery=enable_recovery,
                          record=record).run(start=start)
        if budget is not None:
            result.facts["resources"] = budget.snapshot()
        return result
    finally:
        if own_budget and budget is not None:
            budget.pause()
        # However the run ended -- done, stopped, failed or halted -- nothing it
        # kept for one turn (a builder probe's warm clone) outlives it.
        _run_end_cleanup(ctx.workdir)


def _run_end_cleanup(workdir: Path) -> None:
    """Run the product's run-end cleanups on ``workdir``, however the run ended.

    The walker is core and imports nothing from the product, so a cleanup reaches it
    through the ``refactory.run_end`` entry-point group: the builder probe's warm
    clones (``refactory/refactory_dsh/builder_probe.py``) and the agents' workspaces
    (``refactory/refactory_dsh/workspace.py``). A cleanup that fails is
    logged with its error, which names what it left on disk; it never replaces the
    run's own result or exception. The DSH evaluator calls this too
    (``refactory/refactory_dsh/dsh_runtime.py``). The group is read on every call,
    not cached: a lookup costs under a millisecond, once per run.
    """
    import importlib.metadata

    for entry in importlib.metadata.entry_points(group="refactory.run_end"):
        try:
            entry.load()(Path(workdir))
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "run-end cleanup %s under %s: %s", entry.name, workdir, exc,
            )
