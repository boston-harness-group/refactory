"""Run one refactory node, from the command line.

    python -m refactory_dsh.run <module>:<function>

The runtime hands a node its own declaration under ``DSH_WORKFLOW_NODE`` and knows
nothing about the graph that invoked it, so this runner builds the same context the
walking executor builds and invokes the same callable. A deterministic node's whole
contract is: read the files you declared, write the files you declared, return the
facts you declared, and (for a gate) print your verdict.

The verdict is the **last non-empty stdout line**, which is how the spine routes a
gate. A gate that says why (a :class:`~refactory_dsh.nn.Verdict` with a reason) has
the reason printed on the lines before it, and the spine hands that to the node a
reloop goes back to and to the CEO. So a gate logs to stderr: anything else it
printed to stdout would be read as part of its reason.

Exit codes
----------
- 0: success.
- 1: general node failure (any exception except ``BuildLoopStop``).
- 2: usage error.
- 3: build-loop limit stop (``BuildLoopStop``).  The spine maps this to
  status ``failed`` when the ``--- refactory failure ---`` marker is present
  in stderr.  ``format_failure_detail`` prints reason-first text after the
  marker so the spine keeps the reason, not the tail.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from refactory_dsh.lowering import DECLARATIONS, KNOBS_ENV
from refactory_dsh.nn import RunContext
from refactory_dsh.nn.resolve import resolve

#: Where the facts a run produced are collected, so the Python side can score a run
#: that DSH executed. This file is the runtime boundary.
FACTS = "facts.json"

#: Taken while a node writes the facts it produced. A fork's branches run at once on
#: the spine -- the check beside its reviews, for one -- and each reads the facts when
#: it starts and writes them when it finishes, so each writes back only what it changed.
FACTS_LOCK = ".refactory/facts.lock"

#: The spine passes each node its own declaration under this name.
NODE_ENV = "DSH_WORKFLOW_NODE"

#: The working directory the command runs in, if the runtime says so explicitly.
WORKDIR_ENV = "REFACTORY_WORKDIR"

#: The most of a gate's reason the runner prints, marker included. It is the spine's
#: own limit (``REASON_LIMIT`` in workflow-spine's ``execute.ts``): a whole check log
#: printed ahead of the verdict could crowd the verdict out of the stdout the shell
#: captures, and the spine would route on a line of the log.
REASON_LIMIT = 4_000


def _clipped(reason: str) -> str:
    """``reason``, keeping its head and its tail once it is over :data:`REASON_LIMIT`."""
    if len(reason) <= REASON_LIMIT:
        return reason
    marker = f"\n… (clipped from {len(reason)} characters) …\n"
    half = (REASON_LIMIT - len(marker)) // 2
    return reason[:half] + marker + reason[-half:]


#: The marker ``refactory_dsh.run`` prints before the failure detail so the spine
#: can locate the structured block even when earlier stderr lines push it down.
FAILURE_MARKER = "--- refactory failure ---"


def format_failure_detail(exc: Exception, limit: int = 2000) -> str:
    """Build the failure text the spine keeps, reason first.

    Layout within *limit* characters total:

    - ``<Type>: <first line of message>``, clipped to 500 chars (the reason);
    - failing test ids from the message, each prefixed ``FAILED ``;
    - a clipped tail of the rest of the message after its first line.

    When ``<Type>: <message>`` fits within the limit, return it whole.
    """
    msg = str(exc)
    type_name = type(exc).__name__
    full = f"{type_name}: {msg}"
    if len(full) <= limit:
        return full
    first_line = msg.split("\n", 1)[0]
    reason = f"{type_name}: {first_line}"
    if len(reason) > 500:
        reason = reason[:497] + "..."

    from refactory_dsh.base_check import parse_test_failures

    test_ids = parse_test_failures(msg)
    test_block = "\n".join(f"FAILED {tid}" for tid in sorted(test_ids)) if test_ids else ""

    # Build rest: message after the first line
    rest = msg.split("\n", 1)[1] if "\n" in msg else ""

    used = len(reason)
    if test_block:
        used += 1 + len(test_block)  # newline + block
    remaining = limit - used - 1  # 1 for joining newline
    if remaining > 3 and rest:
        tail = "..." + rest[-(remaining - 3):] if len(rest) > remaining else rest
    else:
        tail = ""

    parts = [reason]
    if test_block:
        parts.append(test_block)
    if tail:
        parts.append(tail)
    result = "\n".join(parts)
    return result[:limit]


class BuildLoopStop(ValueError):
    """A build-loop limit reached; no more attempts.

    Raised by the build loop's limits (check exhausted, review reloop,
    authoring unresolved, consecutive refusals) and caught by ``main()``
    which exits with code 3.  The spine maps exit code 3 (with the
    ``--- refactory failure ---`` marker) to status ``failed``.
    """


class RunnerError(RuntimeError):
    """The node could not be run."""


def _manifest() -> dict[str, Any]:
    raw = os.environ.get(NODE_ENV, "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _knobs() -> dict[str, Any]:
    for name in (KNOBS_ENV, "SRF_KNOBS"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _read_facts(workdir: Path) -> dict[str, Any]:
    path = workdir / FACTS
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def edit_facts(workdir: Path, edit: Any) -> None:
    """Apply ``edit`` (a function of the facts dict, which it changes in place) to the
    run's facts as they are on disk now, under :data:`FACTS_LOCK`, and replace the file
    whole, so no reader sees half of it. A facts file that cannot be read is refused
    rather than overwritten: writing back only this node's facts would drop the rest."""
    import fcntl

    lock_path = workdir / FACTS_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    path = workdir / FACTS
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            facts: dict[str, Any] = {}
            if path.exists():
                try:
                    loaded = json.loads(path.read_text())
                except json.JSONDecodeError as exc:
                    raise RunnerError(f"{path} cannot be read ({exc}); refusing to overwrite it") from exc
                if not isinstance(loaded, dict):
                    raise RunnerError(f"{path} is not a JSON object; refusing to overwrite it")
                facts = loaded
            edit(facts)
            temporary = path.with_name(f".{FACTS}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(facts, indent=2, default=str))
            temporary.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def update_facts(workdir: Path, changed: dict[str, Any], removed: tuple[str, ...] = ()) -> None:
    """Set ``changed`` and drop ``removed`` in the run's facts, leaving every other fact as
    it is on disk now: a node that ran beside others must not put back what they wrote
    while it ran (`edit_facts`)."""
    def edit(facts: dict[str, Any]) -> None:
        facts.update(changed)
        for key in removed:
            facts.pop(key, None)

    edit_facts(workdir, edit)


def write_facts(workdir: Path, facts: dict[str, Any]) -> None:
    """Make the run's facts exactly ``facts``, under :data:`FACTS_LOCK`: what the walker
    does, whose facts live in its own memory and reach `facts.json` whole after each node,
    where every reader of a run directory reads them. A facts file that cannot be read is
    replaced, not refused: the walker's facts are the run's, not a merge with it."""
    import fcntl

    lock_path = workdir / FACTS_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    path = workdir / FACTS
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            temporary = path.with_name(f".{FACTS}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(facts, indent=2, default=str))
            temporary.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _write_changes(workdir: Path, before: dict[str, Any], after: dict[str, Any]) -> None:
    """Write back the facts a node changed: those it set or altered, and those it dropped."""
    changed = {key: value for key, value in after.items()
               if key not in before or before[key] != value}
    update_facts(workdir, changed, tuple(key for key in before if key not in after))


def _memory_store(directory: Path) -> Any:
    """The memory store for one node's run, or ``None`` when nothing is declared.

    Roots follow the walker: this run's own directory for ``run`` retention, the
    project's ``.factory`` for ``project``, and the configured global root — so a
    mode that declares a prior gets it on both runtimes rather than on one.
    """
    from refactory_dsh.lowering import read_memory_declarations
    from refactory_dsh.memory import MemoryStore

    declared = read_memory_declarations(directory)
    if not declared:
        return None
    project = Path(os.environ.get("REFACTORY_PROJECT") or project_for_nodes())
    global_root = os.environ.get("REFACTORY_MEMORY_HOME")
    return MemoryStore(
        declared,
        run_root=directory,
        project_root=project / ".factory",
        global_root=Path(global_root) if global_root else None,
    )


def declarations(workdir: Path, node_id: str) -> dict[str, Any]:
    """What the node declared: from the run directory, falling back to the manifest.

    The run directory is authoritative because it carries the facts a node produces,
    which `DSH_WORKFLOW_NODE` has no field for.
    """
    path = Path(workdir) / DECLARATIONS
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict) and isinstance(loaded.get(node_id), dict):
            entry = loaded[node_id]
            result: dict[str, Any] = {
                "reads": list(entry.get("reads") or ()),
                "writes": list(entry.get("writes") or ()),
                "produces": list(entry.get("produces") or ()),
            }
            if entry.get("optional"):
                result["optional"] = True
            return result
    manifest = _manifest()
    if str(manifest.get("id")) == node_id:
        return {
            "reads": list(manifest.get("reads") or ()),
            "writes": list(manifest.get("writes") or ()),
            "produces": [],
        }
    return {"reads": [], "writes": [], "produces": []}


def install_workspace_packages() -> None:
    """Make this project's workspace packages importable in this process.

    A package created into ``.factory/packages`` is importable only where
    ``registry.install_workspace`` has run, and that is the driver's process. This
    runner is the *node's* process — DSH runs every node as its own command — so
    without the same root here an op in a created package cannot be imported, and
    the mode scores nothing on the runtime that ships while working on the walker.
    Idempotent, so an in-process caller that already installed the root pays nothing.
    """
    from refactory_dsh import registry
    from refactory_dsh.paths import project_root, project_for_nodes

    registry.install_workspace(project_for_nodes())


def build_op_context(
    directory: Path,
    node_id: str,
    *,
    knobs: dict[str, Any] | None = None,
    facts: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    model: Any = None,
) -> RunContext:
    """Build the context an op or fn-gate callable receives.

    Both ``run_node()`` (DSH spine) and the walker's child process call this,
    ensuring they build the same context.
    """
    declared = declarations(directory, node_id)
    ctx = RunContext(
        directory,
        knobs=knobs or _knobs(),
        facts=facts or _read_facts(directory),
        task=task or {"node": node_id},
        memory=_memory_store(directory),
        model=model,
    )
    ctx.node_id = node_id
    ctx._declared = tuple(declared["writes"])  # noqa: SLF001
    ctx._declared_reads = tuple(declared["reads"])  # noqa: SLF001
    ctx._declared_facts = tuple(declared["produces"])  # noqa: SLF001
    return ctx


def run_node(
    target: str, *, workdir: Path | None = None, node_id: str | None = None
) -> Any:
    """Invoke one node's callable with the context the runtime provides.

    The node's identity comes from the runtime's manifest, or from an explicit id.
    It must not fall back to the target: the declarations are keyed by node id, so
    guessing the target would silently look up nothing and stop enforcing the
    declarations the node declares.
    """
    install_workspace_packages()
    manifest = _manifest()
    directory = Path(workdir or os.environ.get(WORKDIR_ENV) or project_for_nodes())
    node_id = str(node_id or manifest.get("id") or "")
    if not node_id:
        raise RunnerError(
            f"no node id: set {NODE_ENV} or pass one, because the declarations this "
            "runner enforces are keyed by node id"
        )
    ctx = build_op_context(
        directory, node_id,
        task={"node": node_id, "type": manifest.get("type", "")},
    )
    node_type = manifest.get("type", "")
    is_retryable = node_type not in ("gate", "GateNode")
    started = copy.deepcopy(ctx.facts)
    facts_snapshot = copy.deepcopy(ctx.facts) if is_retryable else None

    def _write_facts() -> None:
        _write_changes(directory, started, ctx.facts)

    try:
        produced = resolve(target)(ctx)
    except Exception:
        if facts_snapshot is None:
            _write_facts()
            raise
        from refactory_dsh.budgets import BudgetExhausted

        # BudgetExhausted / BuildLoopStop: write facts and re-raise without retry.
        # (re-check the exception type since we're in a broad except)
        import sys as _sys
        exc_info = _sys.exc_info()
        if isinstance(exc_info[1], (BudgetExhausted, BuildLoopStop)):
            _write_facts()
            raise
        # Restore facts and retry once.
        ctx.facts = facts_snapshot
        try:
            produced = resolve(target)(ctx)
        except Exception as retry_exc:
            _write_facts()
            # Optional nodes (named in a fork's optional_targets) are recorded
            # as missing instead of failing the run.
            decl = declarations(directory, node_id)
            if decl.get("optional"):
                reviews_dir = directory / "reviews"
                reviews_dir.mkdir(parents=True, exist_ok=True)
                (reviews_dir / f"{node_id}-missing.txt").write_text(
                    f"{type(retry_exc).__name__}: {retry_exc}\n")
                # dm-232: settle the pending delivery marker at the point
                # the branch is recorded missing, so it never goes stale.
                try:
                    from refactory_dsh.agent_delivery import invalidate as _ad_invalidate
                    _ad_invalidate(directory, [node_id], f"optional branch missing: {retry_exc}")
                except Exception as settle_error:
                    # The branch stays missing either way; a marker left pending only
                    # makes the next resume ask for `graph-control acknowledge`.
                    print(f"refactory: run warning: could not settle the delivery marker of "
                          f"optional {node_id}: {type(settle_error).__name__}: {settle_error}; "
                          f"a later resume asks for graph-control acknowledge", file=sys.stderr)
                return {}
            raise
    if isinstance(produced, dict):
        for key, value in produced.items():
            ctx.set_fact(key, value)
    ctx._check_produces()  # noqa: SLF001
    _write_facts()
    return produced


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    node_id = ""
    if "--node" in argv:
        index = argv.index("--node")
        if index + 1 >= len(argv):
            print("--node needs a value", file=sys.stderr)
            return 2
        node_id = argv[index + 1]
        del argv[index : index + 2]
    if len(argv) != 1 or ":" not in argv[0]:
        print(
            "usage: python -m refactory_dsh.run <module.path>:<function> [--node <id>]",
            file=sys.stderr,
        )
        return 2
    try:
        produced = run_node(argv[0], node_id=node_id or None)
    except BuildLoopStop as exc:
        print(FAILURE_MARKER, file=sys.stderr)
        print(format_failure_detail(exc), file=sys.stderr)
        return 3
    except Exception as exc:  # a failing node must fail loudly, and not be a traceback
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(FAILURE_MARKER, file=sys.stderr)
        print(format_failure_detail(exc), file=sys.stderr)
        return 1
    # A gate's output is its reason, then its verdict: the spine routes on the
    # last non-empty line and reports the lines before it as why.
    if isinstance(produced, str) and produced.strip():
        reason = str(getattr(produced, "reason", "")).strip()
        if reason:
            print(_clipped(reason))
        print(produced.strip())
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
