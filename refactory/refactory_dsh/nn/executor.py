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
import importlib
import json
import multiprocessing
import os
import signal
import sys
import threading
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

# Sentinel for SWEEP-6 refilled feedback cleanup.
_SENTINEL = object()

# Gap 20 / correction C: the spine's RUN_WARNING pattern (execute.ts:166).
import re as _re_mod
_RUN_WARNING_RE = _re_mod.compile(r"^(?:[\w.-]+:\s*)?run warning:\s*\S", _re_mod.IGNORECASE)


def _emit_run_warnings(workdir: Path, node_id: str, stderr: str) -> None:
    """Emit ``run_warning`` events for stderr lines that match the spine's pattern.

    The spine (execute.ts:177-178, index.ts:1681-1683) scans an op's stderr for
    lines matching ``/^(?:[\\w.-]+:\\s*)?run warning:\\s*\\S/i`` and journals each as
    a ``run_warning`` event.  The walker captures stderr from child-process ops
    and calls this helper after a successful dispatch.
    """
    if not stderr:
        return
    from refactory_dsh.live import emit as _emit_event
    for line in stderr.split("\n"):
        stripped = line.strip()
        if _RUN_WARNING_RE.match(stripped):
            _emit_event(workdir, "run_warning", node_id=node_id,
                        state="warning", summary=stripped)


# ── forkserver context (module-level, shared across all Executor instances) ──
_forkserver_ctx = multiprocessing.get_context("forkserver")
_forkserver_modules: set[str] = set()
# Lock for thread-safe manipulation of __main__.__file__ during child.start().
# Concurrent fork-branch threads must not race on the process-wide global.
_main_file_lock = threading.Lock()


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


class _BranchFailed(Exception):
    """A branch error string wrapped so it flows through the per-item try/except."""


class _ItemDeadlineError(Exception):
    """An item's lifecycle (setup + branch + verify) exceeded its deadline."""

    def __init__(self, node_id: str, item_id: str, deadline_s: float):
        self.node_id = node_id
        self.item_id = item_id
        self.deadline_s = deadline_s
        super().__init__(
            f"item '{item_id}' of '{node_id}' timed out after {deadline_s:.1f}s"
        )


class ExecutionError(RuntimeError):
    """A run could not be carried out."""


class OpOutcomeUnknown(ExecutionError):
    """A child exited after dispatch without reporting its side effects."""


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

    def _writable(self, name: str) -> Path:
        """The path a write to *name* goes to, cut loose from any run-level input
        a data pass reads through (items.detach_read_through): a pass must not
        change the run's files, whatever the node declares."""
        from refactory_dsh.items import detach_read_through

        target = self.path(name)
        detach_read_through(self.workdir, name)
        return target

    def write_text(self, name: str, text: str) -> Path:
        self._note_write(name)
        target = self._writable(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def append_text(self, name: str, text: str) -> Path:
        self._note_write(name)
        target = self._writable(name)
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

    # ── pass isolation ─────────────────────────────────────────────────────
    def pass_scope(self, pass_dir: Path, produces: tuple[str, ...]):
        """Context manager that isolates one data-node item pass.

        On enter: saves *workdir* and *facts*, sets ``workdir`` to *pass_dir*
        and ``facts`` to a deep copy: a pass that edits a fact in place (appends
        to a list) changes its own copy, as the spine's pass does with its own
        ``facts.json``.

        On exit: merges only keys in *produces* plus ``"item_scores"`` that the
        pass added or changed into the saved (run-level) facts, then restores
        both ``workdir`` and ``facts``.
        """
        import copy
        from contextlib import contextmanager

        @contextmanager
        def _scope():
            saved_workdir = self.workdir
            saved_facts = self.facts
            self.workdir = pass_dir
            self.facts = copy.deepcopy(saved_facts)
            try:
                yield
            finally:
                # Merge only declared produces + item_scores. The pass's copy is
                # discarded on exit, so its values can be handed over as they are.
                allowed = set(produces) | {"item_scores"}
                for key in allowed:
                    if key in self.facts:
                        if key not in saved_facts or saved_facts[key] != self.facts[key]:
                            saved_facts[key] = self.facts[key]
                self.workdir = saved_workdir
                self.facts = saved_facts

        return _scope()


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

    When ``subset.json`` exists in ``workdir``, its ``part`` and ``seed`` are
    forwarded to ``Task.instances()`` so a Task whose items depend on the seed
    sees the evaluation's part and seed.
    """
    items = declared_items(node, workdir)
    if not items and node.task_ref:
        from refactory_dsh.task import load_task, subset_part_seed

        part, seed = subset_part_seed(workdir)
        items = task_items(load_task(node.task_ref, workdir), part=part, seed=seed)
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
            for csv_index, csv_item in enumerate(items):
                csv_item.setdefault("id", str(csv_index))
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


def task_items(
    task: "Task",
    *,
    part: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """A Task's instances as the items a data node iterates.

    ``part`` and ``seed`` reach ``task.instances()`` only when its signature
    accepts them (``task.part_seed_kwargs``), so a Task written without them
    still loads and a TypeError raised inside it is never swallowed.
    """
    from refactory_dsh.task import part_seed_kwargs

    instances = task.instances(**part_seed_kwargs(task.instances, part=part, seed=seed))
    return [
        {"id": instance.id, "path": instance.path, **instance.metadata}
        for instance in instances
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
    """Target for the child process that runs one op or fn gate, and then exits.

    One child per call, so an op that crashes, hangs or leaves state behind cannot
    affect the next one: the interpreter is the parent's as it is at this dispatch.
    The parent kills the child's whole process group at the node's deadline.
    """
    os.setsid()

    def _sigterm_handler(signum: int, frame: Any) -> None:
        raise SystemExit(f"terminated by signal {signum}")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        # Apply the parent's environment snapshot: the forkserver child inherits
        # the forkserver's env (created once), but the parent may have set vars
        # (e.g. via monkeypatch.setenv in tests) that the child needs to see.
        env_snapshot = args_dict.get("env_snapshot")
        if env_snapshot:
            os.environ.clear()
            os.environ.update(env_snapshot)
        # The child imports what an op names on its own: the forkserver started with the
        # parent's sys.path. A workspace package installed since then is picked up here.
        for pkg_name, pkg_paths in args_dict.get("_package_paths", {}).items():
            pkg = importlib.import_module(pkg_name)
            pkg.__path__ = list(pkg_paths)
        importlib.invalidate_caches()
        # SWEEP-3: Run ops in the run directory, matching the spine (execute.ts:204).
        # Set REFACTORY_PROJECT so project_for_nodes() still resolves to the
        # launcher rather than the run directory after the chdir.
        if not os.environ.get("REFACTORY_PROJECT"):
            os.environ["REFACTORY_PROJECT"] = os.getcwd()
        os.chdir(args_dict["workdir"])

        try:
            from refactory_dsh.budgets import ENV_LEDGER, RunBudget

            ledger = args_dict.get("budget_ledger")
            if ledger is None:
                os.environ.pop(ENV_LEDGER, None)
            else:
                os.environ[ENV_LEDGER] = ledger

            # SWEEP-5: ctx.model for ops is None, matching the spine (run.py:314,
            # 344-365) and the documented op API (create_ops.py:126).
            ctx = RunContext(
                Path(args_dict["workdir"]), knobs=args_dict["knobs"],
                facts=args_dict["facts"], task=args_dict["task"],
                memory=args_dict["memory"],
                verdicts=args_dict["verdicts"], enforce_writes=args_dict["enforce_writes"],
            )
            ctx.node_id = args_dict["node_id"]
            ctx.iteration = args_dict["iteration"]
            ctx._declared = tuple(args_dict.get("writes", ()))
            ctx._declared_reads = tuple(args_dict.get("reads", ()))
            ctx._declared_facts = tuple(args_dict.get("produces", ()))

            # Open the same shared ledger the parent holds, so an op that records usage
            # is charged once, in this run (dm-173).
            ctx.budget = RunBudget.from_path(ledger) if ledger else None

            fn = resolve(args_dict["callable_target"])
            # Gap 20 / correction C: capture stderr so run_warning lines can
            # be emitted by the parent, matching the spine (execute.ts:177-178).
            # SWEEP-4: also capture stdout so the parent can extract fn-gate
            # feedback the same way the spine does (execute.ts:148-157).
            import io as _io
            _old_stderr = sys.stderr
            _old_stdout = sys.stdout
            _capture_stderr = _io.StringIO()
            _capture_stdout = _io.StringIO()
            sys.stderr = _capture_stderr
            sys.stdout = _capture_stdout
            try:
                result = fn(ctx)
            finally:
                sys.stderr = _old_stderr
                sys.stdout = _old_stdout
            _captured_stderr = _capture_stderr.getvalue()
            _captured_stdout = _capture_stdout.getvalue()

            conn.send(("ok", {
                "result": result,
                "facts": dict(ctx.facts),
                "writes_made": list(ctx.writes_made),
                "_produced": list(ctx._produced),
                "_stderr": _captured_stderr,
                "_stdout": _captured_stdout,
            }))
        except SystemExit:
            raise  # propagate SIGTERM
        except BaseException as exc:
            # Include any stderr and stdout captured before the exception for
            # error diagnosis: an op that prints and then raises is exactly the
            # run whose diagnostics are wanted.
            try:
                _err_stderr = _capture_stderr.getvalue()
            except Exception:
                _err_stderr = ""
            try:
                _err_stdout = _capture_stdout.getvalue()
            except Exception:
                _err_stdout = ""
            conn.send(("error", {
                "type": type(exc).__qualname__,
                "module": type(exc).__module__,
                "message": str(exc),
                "traceback": "".join(traceback_mod.format_exception(exc)),
                "_stderr": _err_stderr,
                "_stdout": _err_stdout,
            }))
    finally:
        conn.close()


def _finish_op_child(child: Any) -> None:
    """Reap a child that has reported, killing it if it is still there.

    An op that returns with a non-daemon thread still running keeps its child alive,
    and multiprocessing joins every child at interpreter exit: the run would hang after
    the node that caused it. So a child that outlives its grace is killed like a
    timed-out one.
    """
    child.join(timeout=1)
    if child.is_alive():
        _kill_op_child(child)


def _kill_op_child(child: Any) -> None:
    """Stop one op's process group: terminate, grace, then kill whatever is left.

    The grace waits on the group rather than on the child: polling the child would reap
    the leader, freeing its pid, and the kill below names the group by that pid.
    """
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            child.terminate()
        except OSError:
            pass
    grace_until = time.monotonic() + 5
    while time.monotonic() < grace_until:
        try:
            os.killpg(child.pid, 0)  # the group is still there, and so is its id
        except OSError:
            break
        time.sleep(0.05)
    # The leader may have exited on TERM while a descendant ignored it: kill the group
    # anyway, before the join that would free the pid.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        if child.is_alive():
            try:
                child.kill()
            except OSError:
                pass
    child.join(timeout=5)


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
    #: Whether the visit being started is a failed op's one retry (`_try_visit`).
    _retrying: bool = field(default=False, init=False, repr=False)
    _next_after_completion: str | None = field(default=None, init=False, repr=False)
    _retries: set[tuple[str, str, str]] | None = field(default=None, init=False, repr=False)
    _status: str = field(default="", init=False, repr=False)
    #: The run-level work directory, captured before any pass isolation swap.
    _run_workdir: Path | None = field(default=None, init=False, repr=False)
    #: The node a resume starts at, until the walk has entered it: only that data
    #: node keeps the passes it finished before the stop.
    _resume_at: str | None = field(default=None, init=False, repr=False)
    #: Per-item exchanges captured during a data-node pass.
    _item_exchanges: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    #: Per-item records built during the data-node loop.
    _item_records: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
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

        workdir = self._run_workdir or self.ctx.workdir
        path = workdir / PROGRESS
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
                "gate_iterations": dict(self._gate_iterations),
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

        workdir = self._run_workdir or self.ctx.workdir
        names = {name for node in self.graph.nodes.values() for name in (*node.reads, *node.writes)}
        if (workdir / ".refactory/run-config.json").exists():
            names.add(".refactory/run-config.json")
        return {name: fingerprint(workdir / name) for name in sorted(names)}

    def _write_status(self, status: str, nodes: list[str], *, error: str | None = None,
                      failed: str | None = None) -> None:
        """Write the run's facts and its status where the spine writes them (`facts.json`,
        `.refactory/spine-status.json`), so a walked run reads as a driven one does.
        Best effort, as the progress record is.

        ASK-4b / gap 20: Also emit a ``graph_state`` event, matching the spine
        (index.ts:1218). Only ``control`` events stay out (gap 19).
        """
        from refactory_dsh.run import write_facts

        workdir = self._run_workdir or self.ctx.workdir
        try:
            write_facts(workdir, self.ctx.facts)
            self.record.write_status(workdir, self.graph, status, nodes, error=error, failed=failed)
            self._status = status
            # Emit graph_state event matching the spine (index.ts:1218).
            from refactory_dsh.live import emit as _emit_event
            _emit_event(self.ctx.workdir, "graph_state", status=status,
                        running_nodes=nodes,
                        **({"error": error} if error else {}),
                        **({"failed_node": failed} if failed else {}))
        except OSError:  # pragma: no cover - an unwritable run directory
            pass

    def _is_retry(self, gate: str, target: str, condition: str) -> bool:
        if self._retries is None:
            self._retries = loop_edges(self.graph)
        return (gate, target, condition) in self._retries

    def run(self, *, start: str | None = None) -> RunResult:
        started = time.time()
        self._run_workdir = self.ctx.workdir
        self._resume_at = start
        visits = self._gate_iterations
        current: str | None = start or self.graph.start or None
        error: str | None = None
        # Track step indices that belong to failed items so the post-walk
        # error check can skip them.
        _failed_item_step_indices: set[int] = set()
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
                    from refactory_dsh.items import make_item_record

                    task = load_task(node.task_ref, self.ctx.workdir) if node.task_ref else None
                except Exception as exc:
                    error = f"{node.id}: {type(exc).__name__}: {exc}"
                    break

                from refactory_dsh.budgets import BudgetExhausted
                _PASSTHROUGH = (BudgetExhausted, KeyboardInterrupt, SystemExit)

                # Per-item command deadline: the same wall the spine's
                # runDataCommand applies (execute.ts:268-278). Derived from
                # the budget's remaining time, defaulting to 3600 s.
                _item_deadline_s = self._data_item_deadline()

                # Ask 4: SIGALRM can only be armed on the main thread.
                # Off the main thread, skip arming and emit a run_warning
                # event so the unarmed deadline is recorded in the run's
                # events (the same channel the spine uses for warnings).
                _on_main_thread = threading.current_thread() is threading.main_thread()
                if _item_deadline_s is not None and not _on_main_thread:
                    _item_deadline_s = None  # disable deadline
                    try:
                        from refactory_dsh import live
                        live.emit(
                            self.ctx.workdir, "run_warning",
                            node_id=node.id, state="warning",
                            summary=(
                                "per-item deadline is not armed because the "
                                "executor is not on the main thread"
                            ),
                        )
                    except Exception:
                        pass  # best-effort warning

                # Scale step cap for data nodes: save the initial step count
                # and the original max_steps so we can account for the kept
                # history after the loop. Scale max_steps by item count so
                # large benchmarks don't hit the global cap.
                _data_steps_start = len(self.steps)
                _subgraph_node_count = len([
                    n for n in self.graph.nodes.values()
                    if n.id != node.id
                ]) or 1
                _original_max_steps = self.max_steps
                self.max_steps += (len(items) - 1) * _subgraph_node_count

                # Read part/seed from subset.json once for all items (ask 5).
                from refactory_dsh.task import part_seed_kwargs, subset_part_seed
                _walker_part, _walker_seed = subset_part_seed(self.ctx.workdir)
                _setup_kwargs: dict[str, Any] = {}
                _verify_kwargs: dict[str, Any] = {}
                if task is not None:
                    _setup_kwargs = part_seed_kwargs(
                        task.setup, part=_walker_part, seed=_walker_seed)
                    _verify_kwargs = part_seed_kwargs(
                        task.verify, part=_walker_part, seed=_walker_seed)

                scores: list[dict[str, Any]] = []
                _data_halted = False

                # ── pass isolation: compute declared reads and prepare ──
                from refactory_dsh.items import (
                    finished_pass_record, mark_pass_complete, pass_dir_of,
                    pass_read_through, prepare_pass_dir,
                )
                import shutil as _shutil
                _run_workdir = self._run_workdir or self.ctx.workdir
                # The run-level files the pass reads through: what the pass's own
                # nodes read and do not write -- the same function, over the same
                # node set, as the spine's loader (items._subgraph_declared_reads).
                _declared_reads = pass_read_through(
                    {n.id: (n.reads, n.writes) for n in self.graph.nodes.values()},
                    self.graph.data_body(node.id), node.writes,
                )
                # Only a resume of this very node goes on from its finished
                # passes; any other entry starts the node clean, as the spine's
                # first `begin` does (items.command_begin, index 0).
                _passes_root = _run_workdir / "passes" / node.id
                _resuming = self._resume_at == node.id
                self._resume_at = None
                if not _resuming and _passes_root.exists():
                    _shutil.rmtree(_passes_root)

                for index, item in enumerate(items):
                    # Resume: an item whose pass finished is not run again; its
                    # record -- kept with the pass when it finished -- is.
                    if _resuming:
                        _done_dir = pass_dir_of(
                            _run_workdir, node.id, str(item.get("id") or index), index,
                        )
                        if (_done_dir / ".done").exists():
                            _kept = finished_pass_record(_done_dir)
                            if _kept is not None:
                                scores.append(_kept)
                                self._item_records.append(_kept)
                            continue
                    _item_steps_start = len(self.steps)

                    instance = instance_of(item)
                    item_id = str(item.get("id") or index)

                    # Create the isolated pass directory.
                    _pass_dir = prepare_pass_dir(
                        _run_workdir, node.id, item_id, index,
                        declared_reads=_declared_reads,
                    )
                    _pass_completed = False
                    record = None

                    _alarm_armed = False
                    _prev_handler = None
                    _prev_alarm = 0
                    with self.ctx.pass_scope(_pass_dir, node.produces):
                      try:
                        # Phase 1: arm deadline around setup + item loading only.
                        if _item_deadline_s is not None:
                            def _deadline_handler(signum: int, frame: Any) -> None:
                                raise _ItemDeadlineError(node.id, item_id, _item_deadline_s)  # type: ignore[arg-type]
                            _prev_handler = signal.signal(signal.SIGALRM, _deadline_handler)
                            _prev_alarm = signal.alarm(int(max(1, _item_deadline_s)))
                            _alarm_armed = True

                        if task is not None:
                            task.setup(instance, self.ctx.workdir, **_setup_kwargs)
                        # The item is written as the data node, so the declaration that
                        # governs it is the data node's — not the last node of the pass
                        # before it.
                        self.ctx.node_id = node.id
                        self.ctx._declared = node.writes  # noqa: SLF001
                        self.ctx._declared_reads = node.reads  # noqa: SLF001
                        self.ctx._declared_facts = node.produces  # noqa: SLF001
                        self.ctx._produced = set()  # noqa: SLF001
                        self.ctx.write_json(CURRENT_ITEM, item)
                        # Dual-write current_item to run-level workdir for progress (PIT-005).
                        (_run_workdir / CURRENT_ITEM).write_text(
                            json.dumps(item, indent=2, default=str)
                        )
                        # Clear per-item exchanges before each pass.
                        self._item_exchanges = []

                        # Disarm deadline before walking the branch.
                        if _alarm_armed:
                            signal.alarm(0)
                            _alarm_armed = False

                        # Phase 2: walk the branch (no deadline active).
                        branch_error = self._walk_branch(
                            node.subgraph_entry, node.subgraph_exit, include_barrier=True, iterations={}
                        )
                        if branch_error is not None:
                            raise _BranchFailed(branch_error)

                        # Phase 3: arm deadline around verify only.
                        if _item_deadline_s is not None:
                            signal.signal(signal.SIGALRM, _deadline_handler)
                            signal.alarm(int(max(1, _item_deadline_s)))
                            _alarm_armed = True

                        if task is not None:
                            verdict = task.verify(instance, self.ctx.workdir, **_verify_kwargs)
                        else:
                            verdict = None
                      except _PASSTHROUGH:
                        raise
                      except HaltStop as exc:
                        # Terminal halt ends the data node.
                        error = str(exc)
                        self._record(node.id, error=error, outcome="halted")
                        _data_halted = True
                        break
                      except GateStop as exc:
                        # A user gate in a data pass must pause the run
                        # (re-raise like a halt), not become a failed item.
                        self._record(node.id, error=str(exc), outcome="refused")
                        error = str(exc)
                        _data_halted = True  # break out of both loops
                        break
                      except _ItemDeadlineError as exc:
                        # Deadline exceeded: record as timeout, not retryable.
                        metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
                        record = make_item_record(
                            item_id, None, error=exc,
                            exchanges=list(self._item_exchanges),
                            label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                            label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                            metadata=metadata,
                        )
                        if record is not None:
                            record["error_kind"] = "timeout"
                            scores.append(record)
                            self._item_records.append(record)
                        _failed_item_step_indices.update(
                            range(_item_steps_start, len(self.steps))
                        )
                        _pass_completed = True
                        continue
                      except _BranchFailed as exc:
                        # Wrap in RuntimeError so error_kind matches the
                        # spine's "RuntimeError" for subgraph failures.
                        metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
                        record = make_item_record(
                            item_id, None, error=RuntimeError(str(exc)),
                            exchanges=list(self._item_exchanges),
                            fail_reason=str(exc),
                            label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                            label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                            metadata=metadata,
                        )
                        if record is not None:
                            scores.append(record)
                            self._item_records.append(record)
                        _failed_item_step_indices.update(
                            range(_item_steps_start, len(self.steps))
                        )
                        _pass_completed = True
                        continue
                      except Exception as exc:
                        metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
                        record = make_item_record(
                            item_id, None, error=exc,
                            exchanges=list(self._item_exchanges),
                            label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                            label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                            metadata=metadata,
                        )
                        if record is not None:
                            scores.append(record)
                            self._item_records.append(record)
                        _failed_item_step_indices.update(
                            range(_item_steps_start, len(self.steps))
                        )
                        _pass_completed = True
                        continue
                      else:
                        metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
                        # Prediction comes from VerifyResult.prediction whenever
                        # the Task sets it (both pass and fail).  Fall back to
                        # verdict.details on pass and "" on fail only when the
                        # Task gives none.
                        _prediction: str | None = None
                        if verdict is not None:
                            if verdict.prediction:
                                _prediction = verdict.prediction
                            elif verdict.passed:
                                _prediction = verdict.details
                            else:
                                _prediction = ""
                        record = make_item_record(
                            item_id, verdict,
                            exchanges=list(self._item_exchanges),
                            prediction=_prediction,
                            label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
                            label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
                            fail_reason=verdict.details if verdict is not None and not verdict.passed and verdict.details else None,
                            metadata=metadata,
                        )
                        if record is not None:
                            scores.append(record)
                            self._item_records.append(record)
                        _pass_completed = True
                      finally:
                        # Disarm the per-item deadline and restore the
                        # previous handler/alarm.  The guard is on
                        # _prev_handler (captured when arming), NOT on
                        # _alarm_armed, so restoration happens even when the
                        # alarm was disarmed before the branch walk (Ask 4).
                        if _alarm_armed:
                            signal.alarm(0)
                        if _prev_handler is not None:
                            signal.signal(signal.SIGALRM, _prev_handler)
                        if _prev_alarm > 0:
                            signal.alarm(_prev_alarm)
                        if _pass_completed:
                            mark_pass_complete(_pass_dir, record)

                # Restore max_steps, offset by the steps the data loop added
                # so the main walk loop's cap accounts for the kept history.
                _data_steps_added = len(self.steps) - _data_steps_start
                self.max_steps = _original_max_steps + _data_steps_added

                if _data_halted:
                    break
                # All-items-failed rule: the data node fails only when every
                # item fails. Count against total items, not records.
                if scores:
                    failed_count = sum(
                        1 for r in scores if r.get("error_text") is not None
                    )
                    if "item_scores" in node.produces:
                        self.ctx.set_fact("item_scores", scores)
                    if failed_count == len(items):
                        error = f"{node.id}: all {len(items)} items failed"
                        break
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
                        isinstance(exc, (BudgetExhausted, CancelledError, subprocess.TimeoutExpired, OpOutcomeUnknown))
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
            for step_idx, step in enumerate(self.steps):
                if step.error and step_idx not in _failed_item_step_indices:
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
            item_records=list(self._item_records),
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

        Nested forks and joins within a branch are handled inline: when a fork is
        encountered, ``run_fork`` is called recursively; when a join is encountered,
        it is recorded and the walk continues to the next node.
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
            # Nested fork inside a branch: delegate to run_fork.
            if node.kind == "fork":
                from .fork import run_fork

                self.steps.append(Step(node.id, node.kind, 0.0))
                self.record.complete(node)
                try:
                    error = run_fork(self, node)
                except GateStop as exc:
                    return str(exc)
                except HaltStop:
                    raise
                except Exception as exc:
                    return f"{node.id}: {type(exc).__name__}: {exc}"
                if error is not None:
                    return error
                current = self.graph.out_edges(node.id)[0].target
                continue
            # Nested join inside a branch: step over it.
            if node.kind == "join":
                self.steps.append(Step(node.id, node.kind, 0.0))
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
                if self._missing_optional(node, exc):
                    # Only the optional node is optional: it is recorded failed and
                    # missing, and its branch goes on with its next node, as on the spine.
                    error = f"{type(exc).__name__}: {exc}"
                    self.steps.append(Step(node.id, node.kind, time.time() - started,
                                           reason=f"missing: {error}"))
                    self.record.complete(node)
                    out = self.graph.out_edges(node.id)
                    current = out[0].target if out else None
                    continue
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
            # SWEEP-8: record as halted, not failed, matching the spine
            # (spine.ts:972-980, 1003-1006).
            raise HaltStop(
                node.id,
                reason=f"max_iterations of {node.max_iterations}; "
                f"retry {verdict!r} was refused before dispatching {target!r}",
            )

    # ── retry wrapper ──────────────────────────────────────────────────────
    def _missing_optional(self, node: Node, exc: BaseException) -> bool:
        """Whether ``node``'s failure makes it missing rather than failing its branch.

        A node a fork names in ``optional_targets`` is optional, and only that node:
        an op or agent that failed is recorded missing (``reviews/<node>-missing.txt``,
        its delivery marker settled), as the spine's runner and worker record it, and its
        branch continues. A budget or build-loop stop, a timeout or an unknown outcome
        still fails it, as they do on the spine.
        """
        from refactory_dsh.budgets import BudgetExhausted
        from refactory_dsh.run import BuildLoopStop

        from .fork import _record_missing, _settle_missing

        optional = any(node.id in (fork.optional_targets or ())
                       for fork in self.graph.nodes.values() if fork.kind == "fork")
        if not optional or node.kind not in ("op", "agent") or isinstance(
                exc, (BudgetExhausted, BuildLoopStop, OpTimeoutError, OpOutcomeUnknown)):
            return False
        error = f"{type(exc).__name__}: {exc}"
        _record_missing(self.ctx.workdir, node.id, error)
        _settle_missing(self.ctx.workdir, [node], error)
        return True

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
        except (OpTimeoutError, OpOutcomeUnknown):
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
            self._retrying = True
            try:
                result = self._visit(node)
            finally:
                self._retrying = False

            # Record the successful retry.
            self.steps.append(
                Step(node.id, node.kind, time.time() - step_started,
                     verdict="retried", error=None)
            )
            return result

    # ── effective timeout ──────────────────────────────────────────────────
    def _effective_timeout(self, node: Node) -> float | None:
        """Derive per-op timeout from the shared formula and clamp to budget.

        Every op and fn gate runs in a reused forkserver child so the
        run-dir CWD, ``ctx.model=None`` and stdout/stderr capture are
        per-process.  No ``os.chdir`` or ``sys.std*`` swaps in the parent.

        Classified ops (smoke, check, deliver) use the shared timeout formula
        from ``lowering.command_timeout_s()``.  Unclassified ops and fn gates
        receive a 3600-second default, as both shipped spine configurations set
        (``packages/bundle/refactory/presets/ceo.patch.yml`` and
        ``packages/bundle/graph/cordis.patch.yml`` each pin
        ``commandTimeoutMs: 3600000``; the ``600_000`` in the spine's schema is only
        its fallback when nothing configures one).  All deadlines are clamped to
        the remaining run wall budget (``budget.remaining_s()``) with a 1-second
        floor when ``enable_recovery`` is ``True``.

        An op that exceeds its deadline is killed (SIGTERM → 5 s grace → SIGKILL
        on its process group) and raises ``OpTimeoutError`` with the node id and
        deadline.  Timeouts are never silently retried.
        """
        is_fn_gate = node.kind == "gate" and node.gate_kind == "fn"
        is_op = node.kind == "op"
        # Every op and fn gate runs in a reused forkserver child so the
        # run-dir CWD, ctx.model=None, and stdout/stderr capture are
        # per-process.  No os.chdir or sys.std* swaps in the parent.
        use_forkserver = is_fn_gate or is_op
        if use_forkserver:
            from refactory_dsh.lowering import command_timeout_s as _cmd_timeout_s

            state = {"knobs": {n: {"value": v} for n, v in self.ctx.knobs.items()}}
            raw = _cmd_timeout_s(node, state)
            if raw is None:
                # Unclassified ops get the spine's 3600 s default (dm-173).
                raw = 3600.0
            if self.enable_recovery:
                remaining = self.ctx.budget.remaining_s() if self.ctx.budget is not None else None
                if remaining is not None:
                    raw = min(raw, remaining)
            return max(1.0, float(raw))
        return None

    def _data_item_deadline(self) -> float | None:
        """Per-item command deadline for data nodes, matching the spine.

        The spine's ``runDataCommand`` (execute.ts:268-278) resolves the shell's
        ``timeoutMs`` and clamps it to the remaining budget.  The walker applies
        the same deadline to each item's setup and verify phases (not the
        branch).  The deadline is derived from the budget's remaining time
        with a 3600 s default, matching the spine's ``commandTimeoutMs``
        default.  It does not depend on ``enable_recovery``.
        """
        raw = 3600.0
        remaining = self.ctx.budget.remaining_s() if self.ctx.budget is not None else None
        if remaining is not None:
            raw = min(raw, remaining)
        return max(1.0, float(raw))

    # ── child-process-based op timeout ─────────────────────────────────────
    def _run_op_timed(self, callable_target: str, node: Node, timeout: float) -> Any:
        """Run a callable in a forkserver child process, killed at *timeout*.

        One child per op (``_op_child_target``), never reused: an op that crashes,
        hangs or leaves state behind cannot reach the next one, and the child's
        interpreter is the parent's as it is at this dispatch. The child never
        outlives the dispatch — whether the op reports, times out, dies, or the
        parent is interrupted — so nothing it left running blocks the run's exit.
        A crashed op is never replayed here.
        """
        # SWEEP-5: ctx.model for ops is None, matching the spine (run.py:314,
        # 344-365) and the documented op API (create_ops.py:126). The readiness
        # checks (design_ops.py:302-306) handle None explicitly.
        # Resolve workdir to absolute: the child does os.chdir(workdir) and
        # relative paths would resolve against the new CWD otherwise.
        _abs_workdir = str(self.ctx.workdir.resolve())
        # Forward the current environment snapshot: forkserver children inherit
        # the forkserver's env, not the parent's current one.  Tests that use
        # monkeypatch.setenv need the child to see those changes.
        args_dict: dict[str, Any] = {
            "workdir": _abs_workdir,
            "env_snapshot": dict(os.environ),
            "facts": dict(self.ctx.facts),
            "knobs": dict(self.ctx.knobs),
            "task": dict(self.ctx.task),
            "callable_target": callable_target,
            "node_id": node.id,
            "writes": tuple(node.writes),
            "reads": tuple(node.reads),
            "produces": tuple(node.produces),
            "memory": self.ctx.memory,
            "iteration": self.ctx.iteration,
            "verdicts": dict(self.ctx.verdicts),
            "enforce_writes": self.ctx.enforce_writes,
        }
        # Forward the budget ledger path so the child process can open the
        # same shared RunBudget file (dm-173 / PC-2).
        if self.ctx.budget is not None:
            args_dict["budget_ledger"] = str(self.ctx.budget.path)
        # Forward extended package __path__ entries so the child can import
        # workspace packages that were installed after the forkserver started.
        _pkg = sys.modules.get("refactory_packages")
        if _pkg is not None and hasattr(_pkg, "__path__"):
            args_dict["_package_paths"] = {"refactory_packages": list(_pkg.__path__)}

        _ensure_forkserver_preload(self.graph)
        parent_conn, child_conn = _forkserver_ctx.Pipe()
        try:
            child = _forkserver_ctx.Process(
                target=_op_child_target, args=(child_conn, args_dict))
            # When the parent was launched from stdin (``python -`` or a
            # heredoc), ``__main__.__file__`` is ``<stdin>`` — not a real
            # path.  ``Process.start()`` calls ``get_preparation_data()``
            # which feeds that path to the child's ``_fixup_main_from_path``,
            # which then fails with FileNotFoundError.  Our forkserver child
            # runs ``_op_child_target`` directly and never needs ``__main__``,
            # so we temporarily clear ``__file__`` for the start() call.
            # The lock serialises concurrent fork-branch threads that modify
            # the process-wide ``__main__.__file__`` global.
            import __main__ as _main_mod
            with _main_file_lock:
                _saved_main_file = getattr(_main_mod, "__file__", None)
                _saved_main_spec = getattr(_main_mod, "__spec__", None)
                # Clear both whenever __main__ has them. A __file__ that is not
                # a real file makes _fixup_main_from_path raise; one that IS a
                # real file makes the child re-import the caller's main module,
                # re-running its module-level code once per op visit (a
                # module-level execute() then nests runs on one run directory).
                # __spec__ matters too: get_preparation_data prefers
                # __spec__.name over __file__, so a `python -m` caller re-imports
                # its main module unless this is cleared as well.
                _need_clear = _saved_main_file is not None
                _need_clear_spec = _saved_main_spec is not None
                if _need_clear:
                    _main_mod.__file__ = None  # type: ignore[attr-defined]
                if _need_clear_spec:
                    _main_mod.__spec__ = None  # type: ignore[attr-defined]
                try:
                    child.start()
                finally:
                    if _need_clear:
                        _main_mod.__file__ = _saved_main_file  # type: ignore[attr-defined]
                    if _need_clear_spec:
                        _main_mod.__spec__ = _saved_main_spec  # type: ignore[attr-defined]
        except BaseException:
            parent_conn.close()
            child_conn.close()
            raise
        child_conn.close()

        received = False
        try:
            if not parent_conn.poll(timeout):
                raise OpTimeoutError(node.id, timeout)
            status, payload = parent_conn.recv()
            received = True
        except (EOFError, ConnectionResetError):
            # Its side effects may already have happened. Never replay an
            # unknown outcome as an unrecorded retry.
            raise OpOutcomeUnknown(
                f"op {node.id!r} child exited before reporting its outcome"
            ) from None
        finally:
            parent_conn.close()
            if received:
                # The op has reported. Reap it; kill it only if it lingers, because a
                # non-daemon thread an op started would otherwise block interpreter
                # exit when multiprocessing joins its children.
                _finish_op_child(child)
            else:
                # Timed out, died mid-report, or this frame is unwinding (a stop, a
                # KeyboardInterrupt): nothing of this dispatch outlives it.
                _kill_op_child(child)

        if status == "ok":
            # Merge child state back into the parent's context.
            child_facts = payload.get("facts", {})
            for key, value in child_facts.items():
                if key not in self.ctx.facts or self.ctx.facts[key] != value:
                    self.ctx.facts[key] = value
            self.ctx._produced.update(payload.get("_produced", []))  # noqa: SLF001
            self.ctx.writes_made.extend(payload.get("writes_made", []))
            # Gap 20 / correction C: emit run_warning events for stderr lines
            # matching the spine's RUN_WARNING pattern (execute.ts:166,177-178).
            _child_stderr = payload.get("_stderr", "")
            _emit_run_warnings(self.ctx.workdir, node.id, _child_stderr)
            # Pass child stderr through to the parent's stderr so it remains
            # visible for debugging and test assertions (e.g. capfd).
            if _child_stderr:
                sys.stderr.write(_child_stderr)
            # An op's stdout goes to the parent's STDERR, not its stdout.
            # Ops print diagnostics (design_ops.py's self-check line,
            # repo_change.py's copy notes) and the spine puts those in its
            # command log, so capturing stdout for fn-gate feedback must not
            # swallow them. They cannot go to stdout, though: loop.run() runs
            # the walker in-process and that command's stdout is the single
            # JSON object its caller parses, so an op's print() would corrupt
            # it. stderr keeps the diagnostics and the contract both.
            _child_stdout = payload.get("_stdout", "")
            if _child_stdout:
                sys.stderr.write(_child_stdout)
            result = payload.get("result")
            # SWEEP-4: attach child process stdout to the result so fn-gate
            # feedback extraction in _visit_gate can use it.  A plain str
            # does not accept attribute assignment, so wrap it in Verdict
            # first — this preserves stdout as feedback for fn gates that
            # return a bare string instead of Verdict.
            if _child_stdout and isinstance(result, str):
                from refactory_dsh.nn import Verdict
                if not isinstance(result, Verdict):
                    result = Verdict(result)
                result._fn_stdout = _child_stdout  # type: ignore[attr-defined]
            return result

        # Exception from child: reconstruct and re-raise, preserving stderr.
        return self._reraise_child_error(
            payload, payload.get("_stderr", ""), payload.get("_stdout", ""),
        )

    @staticmethod
    def _reraise_child_error(
        payload: dict[str, Any], stderr: str = "", stdout: str = "",
    ) -> Any:
        """Reconstruct and re-raise an exception sent by the child process.

        The child's stdout is forwarded too, onto the parent's stderr, because
        an op that prints and then raises is exactly the run whose diagnostics
        are wanted. stderr is appended to the exception message as well; stdout
        is only written through.
        """
        if stdout:
            sys.stderr.write(stdout)
        exc_type_name = payload.get("type", "RuntimeError")
        exc_msg = payload.get("message", "")
        exc_tb = payload.get("traceback", "")
        exc_module = payload.get("module", "builtins")

        # Preserve stderr from the child so it is visible in the error.
        if stderr and stderr.strip():
            exc_msg = f"{exc_msg}\nop stderr: {stderr.strip()}"

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
        # The one retry of a failed op (`_try_visit`) is the same visit, as the spine's
        # runner retries inside one command: its start says so.
        retry = {"retry": True} if self._retrying else {}
        emit(self.ctx.workdir, "node_started", node_id=node.id, state="running", summary=render(node.notes, self.ctx),
             **retry)
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
            produced = self._run_op_timed(node.call, node, timeout)
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
        # Refuse an agent node with a tools field, matching the spine's
        # lowering refusal (workers.py:306-307) — SWEEP-7 / gap 15.
        if getattr(node, "tools", None):
            raise ValueError(
                f"agent node {node.id!r} declares a 'tools' field; "
                "tools are not supported on agent nodes"
            )
        from refactory_dsh.agent_delivery import _path, fill_feedback, with_feedback

        # ASK-4a / gap 15: Refuse a node whose declared reads are missing,
        # matching the spine's worker_node.py:82-84.
        for name in node.reads:
            if not _path(self.ctx.workdir, name).exists():
                raise ValueError(f'required worker input is missing: {name}')

        client = self.ctx.model
        if callable(getattr(client, "for_node", None)):
            client = client.for_node(node.id, role=node.role)
        # ── no-tools enforcement ──────────────────────────────────────────
        tool_policy = node.tool_policy or ""
        if callable(getattr(client, "with_tool_policy", None)):
            client = client.with_tool_policy(tool_policy)
        if tool_policy == "none":
            # Validate single-output cardinality before the model
            if len(node.writes) != 1:
                raise ValueError(
                    f"agent node {node.id!r} declares tool_policy='none' but has "
                    f"{len(node.writes)} write(s); a no-tools node must declare "
                    "exactly one output for its captured reply"
                )
            if not callable(getattr(client, "with_tool_policy", None)):
                raise ValueError(
                    f"agent node {node.id!r} declares tool_policy='none' but the "
                    "model client does not support with_tool_policy; cannot "
                    "enforce the policy"
                )
            # Route refusal
            backend = (client.route() if callable(getattr(client, "route", None)) else {}).get("worker_backend", "dsh")
            if backend not in ("dsh", "claude_code"):
                raise ValueError(
                    f"agent node {node.id!r} declares tool_policy='none', but the "
                    f"{backend!r} route cannot enforce it"
                )
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
        # SWEEP-6: Refilled feedback keys are available for rendering but not
        # persisted as run facts, matching the spine (worker_node.py:133-134).
        # Temporarily inject them so render() sees them, then remove after.
        _refilled_snap: dict[str, Any] = {}
        if refilled_keys:
            for k in refilled_keys:
                _refilled_snap[k] = self.ctx.facts.get(k, _SENTINEL)
                self.ctx.facts[k] = values[k]
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
        system = render(node.system_prompt_template, self.ctx) if node.system_prompt_template else ""
        # Gap 20 / correction C: detect unresolved template keys and emit
        # run_warning events, matching the spine (index.ts:1800-1811).
        # Strip the steering block before scanning (agent_delivery.py:838-845).
        import re as _re_tmpl
        from refactory_dsh.steering import STEERING_START, STEERING_END
        _scan = node.prompt_template
        _st_s = _scan.find(STEERING_START)
        _st_e = _scan.find(STEERING_END, _st_s) if _st_s >= 0 else -1
        if _st_s >= 0 and _st_e >= _st_s:
            _scan = _scan[:_st_s] + _scan[_st_e + len(STEERING_END):]
        _tmpl_keys = _re_tmpl.findall(r'\{\{\s*([A-Za-z_]\w*)\s*\}\}', _scan)
        _vals = {**self.ctx.knobs, **self.ctx.facts}
        _unresolved = [k for k in _tmpl_keys if k.strip() not in _vals
                       and k not in (refilled_keys or [])]
        if _unresolved:
            from refactory_dsh.live import emit as _emit_warn
            for _uk in _unresolved:
                _emit_warn(self.ctx.workdir, "run_warning", node_id=node.id,
                           state="warning",
                           summary=f"unresolved template key: {{{{{_uk}}}}}")
        # Pre-flight: refuse a non-empty system prompt on a route that cannot carry it.
        if system:
            from refactory_dsh.workers import capabilities as _caps
            backend = route.get("worker_backend", "dsh")
            if backend in ('dsh', 'codex', 'claude_code', 'openshell') and not _caps(backend).get("system_prompt"):
                raise ValueError(
                    f"agent node {node.id!r} has a system prompt, but the {backend!r} "
                    "route does not support a system-prompt channel"
                )
        if self.ctx.budget is not None and not callable(getattr(client, "with_budget", None)):
            self.ctx.budget.require_usage_capability(False)

        def ask(attempt: Any, sent: str, place: Any) -> str:
            self.model_calls += 1
            if place is None:
                response = _ask_model(attempt, sent, system=system, workdir=self.ctx.workdir)
            else:
                response = workspaces.ask(place, attempt, sent, workdir=self.ctx.workdir, system=system)
            # Exchange-counting rule: one exchange per completed agent call.
            # Retries within a single ask() are transparent to the walker
            # (the model client handles them); the spine path records one
            # exchange per agent_delivery.finish call via record_exchange.
            # Both runtimes therefore count one exchange per completed call.
            # The walker keeps exchanges in memory; the sidecar file is only
            # for the spine path (where separate processes share state via disk).
            self._item_exchanges.append({
                "node": node.id,
                "prompt": sent,
                "response": response,
            })
            return response

        _capture_final = route.get("worker_backend", "dsh") == "dsh"
        if tool_policy == "none":
            _capture_final = True
        try:
            output, _receipt = deliver(
                self.ctx.workdir, node.id, prompt=prompt, writes=list(node.writes),
                reads=list(node.reads), client=client, ask=ask,
                capture_final=_capture_final, rule=node.retry,
                timeout_s=node.timeout_s, budget=self.ctx.budget, workspace=dispatch_workspace,
                tool_policy=tool_policy)
        finally:
            # SWEEP-6: clean up temporarily injected refilled keys, even on error.
            for k, prev in _refilled_snap.items():
                if prev is _SENTINEL:
                    self.ctx.facts.pop(k, None)
                else:
                    self.ctx.facts[k] = prev
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
            # Forkserver child: per-process stdout capture, thread-safe,
            # no global sys.stdout swap.
            verdict = self._run_op_timed(node.evaluator, node, timeout)
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
        # SWEEP-4: Match the spine's feedback text (execute.ts:reasonOf).
        # On the spine, run.py prints Verdict.reason to stdout first, then the
        # verdict label; reasonOf() takes all lines before the last non-empty one.
        # On the walker, stdout capture gets what the fn-gate printed (no label).
        # Combine captured stdout with Verdict.reason, as the spine would see them.
        _captured_stdout = getattr(verdict, '_fn_stdout', '').strip()
        _verdict_reason = str(getattr(verdict, "reason", "") or "")
        if _captured_stdout:
            # Captured stdout is what the fn printed. The spine would also see the
            # reason (from Verdict) and the label. Build the full text as the spine
            # would, then apply reasonOf.
            from refactory_dsh.run import _clipped as _clip_reason
            _full_stdout_parts: list[str] = []
            if _captured_stdout:
                _full_stdout_parts.append(_captured_stdout)
            if _verdict_reason:
                _full_stdout_parts.append(_clip_reason(_verdict_reason))
            _full_stdout_parts.append(str(verdict).strip())
            _full_stdout = '\n'.join(_full_stdout_parts)
            # reasonOf: all lines before the last non-empty one (the verdict label).
            _lines = _full_stdout.split('\n')
            _last = len(_lines) - 1
            while _last >= 0 and _lines[_last].strip() == '':
                _last -= 1
            why = '\n'.join(_lines[:max(_last, 0)]).strip()
        else:
            why = _verdict_reason
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
    omit_steering: bool | None = None,
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
    from refactory_dsh.evaluators import _persist_steering_option
    _persist_steering_option(workdir, omit_steering)
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
    graph = for_run(graph, workdir, omit_steering=omit_steering)
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
    # Read progress.json once, reuse for both RunRecord and gate_iterations (non-core).
    record = RunRecord()
    gate_iterations: dict[str, int] = {}
    if start is not None:
        try:
            _progress = json.loads((workdir / PROGRESS).read_text())
            record = RunRecord.from_progress(_progress)
            gate_iterations = _progress.get("gate_iterations", {})
        except (OSError, ValueError, AttributeError):
            pass
    try:
        executor = Executor(graph, ctx, max_steps=max_steps, enable_recovery=enable_recovery,
                            record=record)
        if gate_iterations:
            executor._gate_iterations.update(gate_iterations)
        result = executor.run(start=start)
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
