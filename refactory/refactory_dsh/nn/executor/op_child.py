"""Running ops in child processes, timeouts, kills."""

from __future__ import annotations

import importlib
import io
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback as traceback_mod
from pathlib import Path
from typing import Any

from refactory_dsh import activity

from ..graph import Graph
from .context import RunContext

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

    record = None
    try:
        # Apply the parent's environment snapshot: the forkserver child inherits
        # the forkserver's env (created once), but the parent may have set vars
        # (e.g. via monkeypatch.setenv in tests) that the child needs to see.
        env_snapshot = args_dict.get("env_snapshot")
        if env_snapshot:
            os.environ.clear()
            os.environ.update(env_snapshot)
        # In a watched run (`activity.WATCH_ENV`) this child, which leads a session of
        # its own, records its group, so a stopped run can end it.
        record = activity.record_process()
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
            ctx.read_renames = dict(args_dict.get("read_renames") or {})
            ctx.snippet = args_dict.get("snippet")

            # Open the same shared ledger the parent holds, so an op that records usage
            # is charged once, in this run (dm-173).
            ctx.budget = RunBudget.from_path(ledger) if ledger else None

            from ..resolve import resolve

            fn = resolve(args_dict["callable_target"])
            # Gap 20 / correction C: capture stderr so run_warning lines can
            # be emitted by the parent, matching the spine (execute.ts:177-178).
            # SWEEP-4: also capture stdout so the parent can extract fn-gate
            # feedback the same way the spine does (execute.ts:148-157).
            _old_stderr = sys.stderr
            _old_stdout = sys.stdout
            _capture_stderr = _MarkedCapture()
            _capture_stdout = _MarkedCapture()
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
        activity.forget_process(record)
        conn.close()


class _MarkedCapture(io.StringIO):
    """An op child's captured output, which a watched run counts as work as it is
    written (`activity.mark`), as the spine's node runner does."""

    def write(self, text: str) -> int:
        activity.mark()
        return super().write(text)


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
