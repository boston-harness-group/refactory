"""The per-item half of a data node, from the command line.

    python -m refactory_dsh.items list  --node <id>
    python -m refactory_dsh.items begin --node <id> --index <n>
    python -m refactory_dsh.items end   --node <id> --index <n>

The walking executor runs a data node's whole loop in one process, because it owns
the graph. DSH does not: the subgraph a data node iterates can contain agent nodes,
so the pass has to be driven from outside the graph by the runtime that owns the
model. That splits the loop into three commands, one per fact the runtime cannot
know on its own:

- ``list`` — how many items the selection holds, so the runtime knows how often to
  walk the subgraph. An empty selection is a fact, reported as zero, not an error.
- ``begin`` — materialise one item as ``current_item.json`` and give the Task its
  ``setup`` hook, before the pass reads anything.
- ``end`` — ask the Task's ``verify`` hook what the pass was worth, and fold the
  score into the run's facts.

Both runtimes load items through :func:`refactory_dsh.nn.executor.load_items`, so a
mode means the same thing on either one, including which part of the items the run
directory names. The declarations the IR cannot carry (the node's ``task_ref``,
``source_path``, ``source_format``, ``inline_items`` and ``limit``) travel in the run
directory's declarations file, written by the lowering.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from refactory_dsh.paths import project_root, project_for_nodes
from refactory_dsh.nn import Node
from refactory_dsh.nn.executor import CURRENT_ITEM, ITEMS_OVERRIDE, load_items
from refactory_dsh.task import Task, TaskError, VerifyResult, instance_of, load_task

#: The working directory a node command runs in, when the runtime says so.
WORKDIR_ENV = "REFACTORY_WORKDIR"

#: Where the facts a run produced are collected. The runner writes this per node;
#: the item loop appends to it, so a dataset mode's `item_scores` reaches the same
#: file on either runtime.
FACTS = "facts.json"

#: The fact a data node's per-item verdicts are collected under.
SCORES_FACT = "item_scores"

#: The declaration key the data node's own fields travel under.
DATA_KEY = "data"

#: The key an override uses to mean "every data node of this run".
ANY_NODE = "*"

#: Sidecar file for per-item exchanges within a data-node pass.
#: Written by agent execution (walker or spine commands), read by ``command_end``.
EXCHANGES_SIDECAR = ".refactory/item_exchanges.json"


# ── pass isolation utilities ─────────────────────────────────────────────

#: The longest item-id stem a pass directory name keeps, in UTF-8 bytes. A file
#: name may hold 255 bytes; the stem leaves room for the hash and the index.
_MAX_ID_BYTES = 200

#: Written into every pass directory by :func:`prepare_pass_dir`: it marks the
#: directory as a pass, so a write through one of its read-through links is
#: detached first (:func:`detach_read_through`), on either runtime.
PASS_MARKER = ".refactory/pass.json"

#: The record of a finished item, beside its ``.done`` marker. A resumed walker
#: reads the scores of the items it skips from here.
PASS_RECORD = ".item_record.json"

#: Run-level files every pass reads through, whatever the subgraph declares:
#: the node declarations, the part/seed and item overrides, the person's
#: steering, the run configuration (steering opt-out, pins, driver profile) and
#: the run's worktree record. Links, so a note steered mid-pass is seen live;
#: read-only from a pass, as every read-through is.
PASS_INFRA = (
    ".refactory/declarations.json", "subset.json", "items.json",
    "intent-ledger.md", ".refactory/run-config.json", ".refactory/worktree.json",
)


def sanitize_item_id(item_id: str) -> str:
    """Make an item id safe for use as a directory name.

    Replaces ``/``, ``:``, ``\\``, and ``\\0`` with ``_`` and bounds the stem to
    ``_MAX_ID_BYTES`` bytes.  When either changed the id, a 6-character hex
    digest of the original id is appended for uniqueness.
    """
    sanitized = item_id.replace("/", "_").replace(":", "_").replace("\\", "_").replace("\0", "_")
    encoded = sanitized.encode()
    if len(encoded) > _MAX_ID_BYTES:
        sanitized = encoded[:_MAX_ID_BYTES].decode("utf-8", "ignore")
    if sanitized != item_id:
        suffix = hashlib.sha256(item_id.encode()).hexdigest()[:6]
        sanitized = f"{sanitized}_{suffix}"
    return sanitized


def pass_dir_of(run_dir: Path, node_id: str, item_id: str, index: int) -> Path:
    """Where item *index* of data node *node_id* runs: ``passes/<node>/<id>_<index>``."""
    return Path(run_dir) / "passes" / node_id / f"{sanitize_item_id(item_id)}_{index}"


def pass_read_through(
    declared: dict[str, tuple[tuple[str, ...] | list[str], tuple[str, ...] | list[str]]],
    body: set[str] | list[str] | tuple[str, ...],
    data_writes: tuple[str, ...] | list[str] = (),
) -> tuple[str, ...]:
    """The run-level files a pass reads through: everything its own nodes read.

    *declared* maps node ids to ``(reads, writes)``; *body* names the pass's nodes
    (``Graph.data_body``); *data_writes* are the data node's own writes (the
    current item), which each pass gets fresh and must not link.

    A file the body both reads and writes is still read through when it exists at
    run level: its nodes expect the run's content, as before pass isolation, and a
    write detaches the link into a private copy (`detach_read_through`), so the run's
    file is not changed. A read the run does not have is simply not linked
    (`prepare_pass_dir` links only what exists). One function for both runtimes, so
    their pass directories cannot drift.
    """
    reads: set[str] = set()
    for node_id in body:
        entry = declared.get(node_id)
        if entry is None:
            continue
        reads.update(entry[0])
    return tuple(sorted(r for r in reads if r not in set(data_writes)))


def prepare_pass_dir(
    run_dir: Path,
    node_id: str,
    item_id: str,
    index: int,
    declared_reads: tuple[str, ...] | list[str] = (),
) -> Path:
    """Create and return the pass directory for one item.

    ``<run_dir>/passes/<node_id>/<sanitized_item_id>_<index>/``

    * Copies ``facts.json`` from *run_dir* into the pass directory (seeding the
      fact scope for the spine path).
    * Links :data:`PASS_INFRA` and each path in *declared_reads* that exists at
      *run_dir* and not yet in the pass, so subgraph nodes read them in place.
      A write to one of them is detached from the run first
      (:func:`detach_read_through`), so a pass cannot change a run-level input.
    """
    run_dir = Path(run_dir)
    pass_dir = pass_dir_of(run_dir, node_id, item_id, index)
    pass_dir.mkdir(parents=True, exist_ok=True)

    # Seed the pass's fact scope from the run-level facts.
    facts_src = run_dir / FACTS
    facts_dst = pass_dir / FACTS
    if facts_src.exists() and not facts_dst.exists():
        shutil.copy2(str(facts_src), str(facts_dst))

    for name in (*PASS_INFRA, *declared_reads):
        src = run_dir / name
        dst = pass_dir / name
        if src.exists() and not dst.exists() and not dst.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())

    marker = pass_dir / PASS_MARKER
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"node": node_id, "item": item_id, "index": index}) + "\n")
    return pass_dir


def detach_read_through(workdir: Path, name: str) -> None:
    """Before a write to *name* in a pass directory, cut it loose from the run.

    A pass reads run-level inputs through links (:func:`prepare_pass_dir`).
    Writing through one would change the run's file for every later item and the
    rest of the run, so each linked component of *name* is replaced first: a
    linked file by a copy of itself (an append keeps the content), a linked
    directory by a real directory linking its entries. Outside a pass directory
    (no :data:`PASS_MARKER`) this does nothing: a run's own links are the run's
    business.
    """
    workdir = Path(workdir)
    if not (workdir / PASS_MARKER).exists():
        return
    current = workdir
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink():
            source = current.resolve()
            current.unlink()
            if source.is_dir():
                current.mkdir()
                for child in source.iterdir():
                    (current / child.name).symlink_to(child)
            elif source.exists():
                shutil.copy2(str(source), str(current))
        if not current.exists():
            return


def mark_pass_complete(pass_dir: Path, record: dict[str, Any] | None = None) -> None:
    """Finish a pass: keep its item record (when there is one), then the ``.done`` marker.

    The record is written first, so a ``.done`` pass always has the score a
    resume needs (an interrupted write leaves an unfinished pass, rerun).
    """
    if record is not None:
        pending = pass_dir / f"{PASS_RECORD}.tmp"
        pending.write_text(json.dumps(record, indent=2, default=str))
        os.replace(pending, pass_dir / PASS_RECORD)
    (pass_dir / ".done").write_text("")


def finished_pass_record(pass_dir: Path) -> dict[str, Any] | None:
    """The record a finished pass kept, or None when it is unfinished or kept none."""
    if not (pass_dir / ".done").exists():
        return None
    try:
        record = json.loads((pass_dir / PASS_RECORD).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def merge_pass_results(
    run_dir: Path,
    pass_dir: Path,
    produces: tuple[str, ...],
) -> None:
    """Merge declared produces from a pass directory back to the run.

    Only keys listed in *produces* plus ``item_scores`` are merged from the
    pass's ``facts.json`` into the run's ``facts.json``.  Everything else stays
    in the pass directory.
    """
    facts_path = pass_dir / FACTS
    if not facts_path.exists():
        return
    try:
        pass_facts = json.loads(facts_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(pass_facts, dict):
        return
    allowed = set(produces) | {SCORES_FACT}
    for key in allowed:
        if key not in pass_facts:
            continue
        value = pass_facts[key]
        if key == SCORES_FACT and isinstance(value, list):
            _write_facts(run_dir, key, value)
        else:
            from refactory_dsh.run import edit_facts

            def _set(facts: dict[str, Any], k: str = key, v: Any = value) -> None:
                facts[k] = v

            edit_facts(run_dir, _set)


def _subgraph_declared_reads(workdir: Path, node: Node) -> tuple[str, ...]:
    """The run-level files *node*'s passes read through, from the run's declarations.

    The lowering writes the pass's own nodes beside the data node
    (``subgraph_nodes``, from ``Graph.data_body``); :func:`pass_read_through` keeps
    what they read and do not write. Declarations lowered before that key existed
    name no body: every other node's reads are taken then, with only the data
    node's own writes left out (an upstream file is still read through; nothing
    is wrongly withheld).
    """
    try:
        declarations = _declarations(workdir)
    except Exception:
        return ()
    entry = declarations.get(node.id)
    body = entry.get("subgraph_nodes") if isinstance(entry, dict) else None
    declared: dict[str, tuple[list[str], list[str]]] = {}
    for nid, decl in declarations.items():
        if nid == node.id or not isinstance(decl, dict):
            continue
        writes = list(decl.get("writes") or []) if isinstance(body, list) else []
        declared[nid] = (list(decl.get("reads") or []), writes)
    return pass_read_through(declared, body if isinstance(body, list) else set(declared), node.writes)


def record_exchange(workdir: Path, node: str, prompt: str, response: str) -> None:
    """Append one ``{node, prompt, response}`` exchange to the per-item sidecar.

    Called from agent execution in both runtimes so that ``command_end`` can
    include per-node exchanges in the item record.
    """
    path = workdir / EXCHANGES_SIDECAR
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    existing.append({"node": node, "prompt": prompt, "response": response})
    path.write_text(json.dumps(existing, default=str))


def _read_and_clear_exchanges(workdir: Path) -> list[dict[str, Any]]:
    """Read per-item exchanges from the sidecar, then remove the file."""
    path = workdir / EXCHANGES_SIDECAR
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        path.unlink(missing_ok=True)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def make_item_record(
    item_id: str,
    verdict: VerifyResult | None,
    error: Exception | None = None,
    *,
    exchanges: list[dict[str, Any]] | None = None,
    prediction: str | None = None,
    label: str | None = None,
    label_text: str | None = None,
    fail_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a per-item record for one evaluated item.

    One record per item, same shape on both runtimes.  Extends F4's existing
    error_kind convention (exception class name or ``"timeout"``).  The
    ``exchanges`` list carries one ``{node, prompt, response}`` dict per agent
    call in the item's pass, so a multi-node pass records every model call.

    **Success path** (error is None, verdict is not None): ``{"id", "passed", "score"}``.
    **Failure path** (error is not None): ``{"id", "passed": False, "score": 0.0,
    "error_text", "error_kind"}``.
    **No-verdict path** (error is None, verdict is None): ``None`` — the Task chose
    not to grade this item.
    """
    base: dict[str, Any] = {
        "exchanges": list(exchanges or []),
        "prediction": prediction,
        "label": label,
        "label_text": label_text,
        "fail_reason": fail_reason,
        "metadata": dict(metadata or {}),
    }
    if error is not None:
        return {
            "id": item_id,
            "passed": False,
            "score": 0.0,
            "error_text": str(error)[:4096],
            "error_kind": type(error).__name__,
            **base,
        }
    if verdict is None:
        return None
    return {
        "id": item_id,
        "passed": verdict.passed,
        "score": verdict.score,
        "error_text": None,
        "error_kind": None,
        **base,
    }


def source_override(workdir: Path, node_id: str) -> dict[str, Any]:
    """What this run says the data node should read, or an empty answer.

    The mode's graph declares the source it was built with; a run may name another,
    which is what makes a harness for a class of work usable on items it has never
    seen without editing the mode. An entry for the node wins over the ``*`` entry, so
    a run can point one node somewhere specific and the rest somewhere general.
    """
    path = workdir / ITEMS_OVERRIDE
    if not path.exists():
        return {}
    try:
        declared = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(declared, dict):
        return {}
    entry = declared.get(node_id, declared.get(ANY_NODE, {}))
    return dict(entry) if isinstance(entry, dict) else {}


def write_source(
    workdir: Path, *, source_path: str, source_format: str = "directory", node_id: str = ANY_NODE,
) -> Path:
    """Point a data node of this run at a source, before the graph starts.

    Written whole, because the override is the run's answer for every node it names:
    a caller that points one node at a directory is not also asking to keep a path
    another caller wrote for a different node.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / ITEMS_OVERRIDE
    declared: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            declared = loaded
    declared[node_id] = {"source_path": str(source_path), "source_format": str(source_format)}
    path.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n")
    return path


#: Exit code used when the all-fail rule fires. The spine distinguishes this
#: from other ``end`` failures so it can propagate the all-fail rule as a node
#: failure while recording other non-zero exits as failed items (Ask 3).
ALL_FAIL_EXIT_CODE = 7


class ItemsError(RuntimeError):
    """A data node cannot be iterated: named, because the node id alone is not a reason."""


class AllItemsFailedError(ItemsError):
    """Raised when every item in a data pass failed (the all-fail rule)."""


def _declarations(workdir: Path) -> dict[str, Any]:
    from refactory_dsh.lowering import read_declarations

    return read_declarations(workdir)


def data_node(workdir: Path, node_id: str) -> Node:
    """The data node the declarations describe, as the graph language's own node.

    Rebuilding the node rather than re-deriving its fields from the manifest keeps
    one reader of the declarations: the manifest is what the IR can carry, and the
    data node's source is exactly what it cannot.
    """
    entry = _declarations(workdir).get(node_id)
    if not isinstance(entry, dict):
        raise ItemsError(
            f"node {node_id!r} is not declared in the run directory {workdir}; the "
            "runner writes the declarations before it starts a node, so an absent "
            "entry means the node id does not name a node of this run"
        )
    if entry.get("kind") != "data":
        raise ItemsError(
            f"node {node_id!r} is declared as {entry.get('kind')!r}, not 'data'; only "
            "a data node iterates items"
        )
    fields = entry.get(DATA_KEY)
    if not isinstance(fields, dict):
        raise ItemsError(
            f"node {node_id!r} is a data node but its source is not declared "
            f"(no {DATA_KEY!r} entry); re-lower the mode with this version of refactory"
        )
    return Node(
        node_id,
        "data",
        reads=tuple(entry.get("reads") or ()),
        writes=tuple(entry.get("writes") or ()),
        produces=tuple(entry.get("produces") or ()),
        **fields,
    )


def items_of(node: Node, workdir: Path) -> list[dict[str, Any]]:
    """The data node's items, loaded the one way both runtimes load them.

    A node bound to a Task and naming no source takes its items from the Task, which
    is what makes ``task_ref`` a binding rather than a line in a prompt, and the part
    the run directory names is applied there too. Both live in
    :func:`refactory_dsh.nn.executor.load_items` so the walker and DSH cannot select
    different items.
    """
    return load_items(node, workdir)


def _task_of(node: Node, workdir: Path) -> Task | None:
    if not node.task_ref:
        return None
    return load_task(node.task_ref, workdir)


def _write_facts(workdir: Path, fact: str, scores: list[dict[str, Any]]) -> None:
    from refactory_dsh.run import edit_facts

    def add(facts: dict[str, Any]) -> None:
        previous = facts.get(fact)
        if isinstance(previous, list):
            # Replace-by-id merge: new scores supersede previous entries with the
            # same id, preventing double counting when a data node reruns.
            by_id = {str(entry.get("id")): entry for entry in previous if isinstance(entry, dict)}
            for score in scores:
                by_id[str(score.get("id"))] = score
            facts[fact] = list(by_id.values())
        else:
            facts[fact] = list(scores)

    # Under the facts lock, as every node's write is: a data pass can end beside a branch.
    edit_facts(workdir, add)


def command_list(node: Node, workdir: Path) -> dict[str, Any]:
    """How many items the node iterates, and what they are called."""
    items = items_of(node, workdir)
    return {
        "count": len(items),
        "ids": [str(item.get("id") or index) for index, item in enumerate(items)],
    }


def _pass_state_path(workdir: Path, node_id: str) -> Path:
    """Path to the run-local pass state file for a data node."""
    return workdir / ".refactory" / "pass_state" / f"{node_id}.json"


def _read_pass_state(workdir: Path, node_id: str) -> dict[str, int]:
    path = _pass_state_path(workdir, node_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed": 0, "failed": 0}


def _write_pass_state(workdir: Path, node_id: str, state: dict[str, int]) -> None:
    path = _pass_state_path(workdir, node_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def _locked_pass_state_update(
    workdir: Path, node_id: str, updater: Any,  # Callable[[dict], None]
) -> None:
    """Read-modify-write pass_state under an exclusive file lock.

    The lock-file pattern matches ``edit_facts`` (``run.py:178``):
    ``fcntl.flock(lock, fcntl.LOCK_EX)`` around read + modify + write.
    """
    import fcntl

    lock_path = _pass_state_path(workdir, node_id).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = _read_pass_state(workdir, node_id)
            updater(state)
            _write_pass_state(workdir, node_id, state)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def command_begin(node: Node, workdir: Path, index: int) -> dict[str, Any]:
    """Materialise one item and prepare the workspace for its pass."""
    items = items_of(node, workdir)
    if index < 0 or index >= len(items):
        raise ItemsError(
            f"item {index} is out of range: node {node.id!r} selected {len(items)} item(s)"
        )
    item = items[index]
    item_id = str(item.get("id") or index)

    # On the first item, reset pass state, clear stale item_scores, and
    # remove old pass directories (rerun creates fresh ones).
    if index == 0:
        _write_pass_state(workdir, node.id, {"processed": 0, "failed": 0})
        if SCORES_FACT in node.produces:
            from refactory_dsh.run import edit_facts
            edit_facts(workdir, lambda facts: facts.pop(SCORES_FACT, None))
        passes_dir = workdir / "passes" / node.id
        if passes_dir.exists():
            shutil.rmtree(passes_dir)

    # Track processed count in pass state (locked for concurrent items).
    _locked_pass_state_update(
        workdir, node.id,
        lambda ps: ps.__setitem__("processed", ps.get("processed", 0) + 1),
    )

    # Create the isolated pass directory.
    declared_reads = _subgraph_declared_reads(workdir, node)
    pass_dir = prepare_pass_dir(workdir, node.id, item_id, index, declared_reads=declared_reads)

    # Write current_item to both pass_dir (for the subgraph) and workdir
    # (for progress tracking / PIT-005).
    item_json = json.dumps(item, indent=2, default=str)
    (pass_dir / CURRENT_ITEM).write_text(item_json)
    (workdir / CURRENT_ITEM).write_text(item_json)

    # Clear the exchange sidecar in the pass directory.
    sidecar = pass_dir / EXCHANGES_SIDECAR
    if sidecar.exists():
        sidecar.unlink(missing_ok=True)

    task = _task_of(node, workdir)
    if task is not None:
        from refactory_dsh.task import part_seed_kwargs, subset_part_seed
        subset_part, subset_seed = subset_part_seed(workdir)
        try:
            task.setup(
                instance_of(item), pass_dir,
                **part_seed_kwargs(task.setup, part=subset_part, seed=subset_seed),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            record = make_item_record(str(item.get("id") or index), None, error=exc)
            if record is not None and SCORES_FACT in node.produces:
                _write_facts(workdir, SCORES_FACT, [record])
            # Do NOT increment failed here: the subsequent `end --failed`
            # call from the spine will increment it exactly once (Ask 1).
            return {"error": str(exc)}
    return {**item, "pass_dir": str(pass_dir)}


def _existing_item_record(workdir: Path, item_id: str) -> dict[str, Any] | None:
    """Return an existing item_scores record for *item_id*, or None.

    Used to guard against double-writing: when ``command_begin``'s setup-failure
    handler already wrote a record, ``command_end``'s ``--failed`` path should
    not overwrite it (Ask 1).
    """
    path = workdir / FACTS
    if not path.exists():
        return None
    try:
        facts = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(facts, dict):
        return None
    scores = facts.get(SCORES_FACT)
    if not isinstance(scores, list):
        return None
    for entry in scores:
        if isinstance(entry, dict) and str(entry.get("id")) == item_id:
            return entry
    return None


def command_end(
    node: Node, workdir: Path, index: int, *,
    failed_reason: str | None = None,
    failed_kind: str | None = None,
) -> dict[str, Any]:
    """The Task's verdict on one pass, folded into the run's facts.

    When *failed_reason* is set (e.g. a subgraph node exited with code ≠ 0),
    ``task.verify()`` is skipped and the item is recorded as failed — matching
    the walker, which never calls verify after a branch error.

    When *failed_kind* is set (e.g. ``"timeout"``), the ``error_kind`` field in
    the record is overridden, so both runtimes record the same kind.
    """
    items = items_of(node, workdir)
    if index < 0 or index >= len(items):
        raise ItemsError(
            f"item {index} is out of range: node {node.id!r} selected {len(items)} item(s)"
        )
    item = items[index]
    item_id = str(item.get("id") or index)

    # Compute the pass directory (idempotent — reuses the one command_begin created).
    pass_dir = prepare_pass_dir(workdir, node.id, item_id, index)

    # F4 parity: when the caller says the item's subgraph failed, skip verify
    # and write a failure record directly, as the walker does.
    if failed_reason is not None:
        exc = RuntimeError(failed_reason)
        instance = instance_of(item)
        exchanges = _read_and_clear_exchanges(pass_dir)
        metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
        record = make_item_record(
            item_id, None, error=exc,
            exchanges=exchanges,
            label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
            label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
            fail_reason=failed_reason,
            metadata=metadata,
        )
        if record is not None:
            if failed_kind is not None:
                record["error_kind"] = failed_kind
            # Guard: skip writing a second item_scores record when
            # command_begin's setup-failure handler already wrote one for
            # this item id (Ask 1 — preserve the original error_kind).
            if SCORES_FACT in node.produces:
                existing = _existing_item_record(workdir, item_id)
                if existing is None:
                    _write_facts(workdir, SCORES_FACT, [record])
                else:
                    # Use the original record so error_kind stays the
                    # actual exception class (e.g. ValueError), not
                    # RuntimeError from the end --failed wrapper.
                    record = existing
        # Track failed count in pass state (locked for concurrent items).
        _locked_pass_state_update(
            workdir, node.id,
            lambda ps: ps.__setitem__("failed", ps.get("failed", 0) + 1),
        )
        # Merge declared produces from the pass directory to the run directory.
        merge_pass_results(workdir, pass_dir, node.produces)
        mark_pass_complete(pass_dir, record)
        result: dict[str, Any] = {"error": failed_reason, **(record or {})}
        _check_all_items_failed(node, workdir, items, index)
        return result

    task = _task_of(node, workdir)
    if task is None:
        # Merge declared produces even when there is no task.
        merge_pass_results(workdir, pass_dir, node.produces)
        mark_pass_complete(pass_dir)
        _check_all_items_failed(node, workdir, items, index)
        return {}
    instance = instance_of(item)
    # Read per-node exchanges from the pass directory sidecar.
    exchanges = _read_and_clear_exchanges(pass_dir)
    metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
    # Pass part and seed from subset.json to task.verify (ask 5)
    from refactory_dsh.task import part_seed_kwargs, subset_part_seed
    _verify_part, _verify_seed = subset_part_seed(workdir)
    try:
        verdict = task.verify(
            instance, pass_dir,
            **part_seed_kwargs(task.verify, part=_verify_part, seed=_verify_seed),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        record = make_item_record(
            item_id, None, error=exc,
            exchanges=exchanges,
            label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
            label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
            metadata=metadata,
        )
        if record is not None and SCORES_FACT in node.produces:
            _write_facts(workdir, SCORES_FACT, [record])
        # Track failed count in pass state (locked for concurrent items).
        _locked_pass_state_update(
            workdir, node.id,
            lambda ps: ps.__setitem__("failed", ps.get("failed", 0) + 1),
        )
        # Merge declared produces from the pass directory to the run directory.
        merge_pass_results(workdir, pass_dir, node.produces)
        mark_pass_complete(pass_dir, record)
        result = {"error": str(exc), **(record or {})}
        _check_all_items_failed(node, workdir, items, index)
        return result
    if verdict is None:
        # Merge declared produces even with no verdict.
        merge_pass_results(workdir, pass_dir, node.produces)
        mark_pass_complete(pass_dir)
        _check_all_items_failed(node, workdir, items, index)
        return {}
    # Prediction comes from VerifyResult.prediction whenever the Task sets it
    # (both pass and fail).  Fall back to verdict.details on pass and "" on
    # fail only when the Task gives none.
    prediction: str | None
    if verdict.prediction:
        prediction = verdict.prediction
    elif verdict.passed:
        prediction = verdict.details
    else:
        prediction = ""
    record = make_item_record(
        item_id, verdict,
        exchanges=exchanges,
        prediction=prediction,
        label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
        label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
        fail_reason=verdict.details if not verdict.passed and verdict.details else None,
        metadata=metadata,
    )
    if record is not None and SCORES_FACT in node.produces:
        _write_facts(workdir, SCORES_FACT, [record])
    # Merge declared produces from the pass directory to the run directory.
    merge_pass_results(workdir, pass_dir, node.produces)
    mark_pass_complete(pass_dir, record)
    _check_all_items_failed(node, workdir, items, index)
    return record or {}


def _check_all_items_failed(
    node: Node, workdir: Path, items: list[dict[str, Any]], index: int,
) -> None:
    """After the last item's end, fail the node if every item has an error.

    This makes the spine path match the walker's all-items-failed rule: a data
    node fails only when *every* item it processed fails. The check uses
    run-local pass state (processed and failed counts under
    ``.refactory/pass_state/<node_id>.json``), not the ``item_scores`` fact,
    so it works whether or not the node declares ``item_scores`` in its
    ``produces``.
    """
    if index + 1 < len(items):
        return  # not the last item yet
    ps = _read_pass_state(workdir, node.id)
    processed = ps.get("processed", 0)
    failed = ps.get("failed", 0)
    if processed > 0 and failed == processed:
        raise AllItemsFailedError(f"{node.id}: all {processed} items failed")


def command_reset(workdir: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Archive prior facts, then invalidate the facts of explicitly redone work.

    A full dataset restart or explicit graph-control reset archives affected
    node-owned outputs, excluding the restart node's declared inputs and files
    also written by a node outside the invalidated set. Legacy current-item
    corrections keep artifacts unless archiveArtifacts is requested. Pending
    agent snapshots are retired in either case.
    """
    import uuid
    from refactory_dsh.agent_delivery import invalidate

    nodes = request.get('nodes')
    if not isinstance(nodes, list) or not all(isinstance(node, str) for node in nodes):
        raise ItemsError('reset needs the runtime redo plan with a list of node ids')
    declarations = _declarations(workdir)
    if not declarations:
        raise ItemsError('reset requires the run declarations; re-lower this mode before redoing work')
    names = {name for node in nodes for name in declarations.get(node, {}).get('produces', [])}
    # A CEO's redo keeps the counts its gates check against their limits (attempts_used,
    # check_attempts_used, research_attempts_used, repairs_used, ...), so a redo cannot
    # reset a limit; a person's graph-control rerun sends no flag and resets them.
    if request.get('preserveAttemptFacts'):
        names = {n for n in names if not n.endswith(('attempts_used', 'repairs_used'))
                 and n not in ('convergence_extras_used', 'check_failures_history',
                               'check_blocker_history', 'convergence_granted',
                               'consecutive_refusals')}
    path = workdir / FACTS
    prior = path.read_text() if path.exists() else '{}'
    facts = json.loads(prior)
    if not isinstance(facts, dict):
        raise ItemsError('reset requires facts.json to contain an object')
    cleared = sorted(names.intersection(facts))
    for name in cleared:
        facts.pop(name)
    from refactory_dsh.run_artifacts import archive_root

    try:
        archive = archive_root(workdir, 'redo') / uuid.uuid4().hex
    except ValueError as exc:
        raise ItemsError(str(exc)) from exc
    # Replaying the same selected items must not inherit op state such as an
    # attempt counter in a declared output file. Keep explicit restart inputs,
    # and archive by rename so the prior bytes stay inspectable without copies.
    targets: list[Path] = []
    retained: list[str] = []
    if request.get('restartData') or request.get('archiveArtifacts') is True:
        reads = [Path(name) for name in declarations.get(request.get('start'), {}).get('reads', [])]
        writes = {name for node in nodes for name in declarations.get(node, {}).get('writes', [])}
        # A node that is not redone will not recreate what it wrote, so a file it
        # shares with a redone node stays put.
        outside_writes = {name for node_id, decl in declarations.items()
                          if isinstance(decl, dict) and node_id not in nodes
                          for name in decl.get('writes', [])}
        reads.extend(Path(name) for name in writes & outside_writes)
        from refactory_dsh.run_artifacts import owned_outputs

        try:
            targets, retained = owned_outputs(workdir, writes, reads)
        except ValueError as exc:
            raise ItemsError(str(exc)) from exc
    # Clean up pass directories for the invalidated nodes.
    for node_id in nodes:
        passes_dir = workdir / "passes" / node_id
        if passes_dir.exists():
            shutil.rmtree(passes_dir)
    receipt = {**request, 'cleared_facts': cleared,
               'archive': str(archive.relative_to(workdir)),
               'archived_artifacts': [str(path) for path in targets],
               'retained_inputs': retained}
    archive.mkdir(parents=True)
    (archive / FACTS).write_text(prior)
    (archive / 'redo.json').write_text(json.dumps(receipt, indent=2) + '\n')
    temporary = path.with_name('.facts-redo.tmp')
    temporary.write_text(json.dumps(facts, indent=2, default=str) + '\n')
    moved: list[Path] = []
    try:
        for relative in targets:
            destination = archive / 'artifacts' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            (workdir / relative).rename(destination)
            moved.append(relative)
        invalidate(workdir, nodes, f"redo {request.get('requested', '')}")
        temporary.replace(path)
    except Exception:
        # A failed archive/reset must not strand partially moved node outputs.
        for relative in reversed(moved):
            (archive / 'artifacts' / relative).rename(workdir / relative)
        temporary.unlink(missing_ok=True)
        raise
    return {key: receipt[key] for key in ('archive', 'cleared_facts', 'archived_artifacts', 'retained_inputs')}



def _options(argv: list[str]) -> tuple[str, int, str | None, str | None]:
    """The ``--node`` id, ``--index``, optional ``--failed`` reason and ``--failed-kind``."""
    node_id = ""
    index = 0
    failed_reason: str | None = None
    failed_kind: str | None = None
    if "--node" in argv:
        position = argv.index("--node")
        if position + 1 >= len(argv):
            raise ItemsError("--node needs a value")
        node_id = argv[position + 1]
    if "--index" in argv:
        position = argv.index("--index")
        if position + 1 >= len(argv):
            raise ItemsError("--index needs a value")
        try:
            index = int(argv[position + 1])
        except ValueError as exc:
            raise ItemsError(f"--index takes an item number, not {argv[position + 1]!r}") from exc
    if "--failed" in argv:
        position = argv.index("--failed")
        if position + 1 >= len(argv):
            raise ItemsError("--failed needs a value")
        failed_reason = argv[position + 1]
    if "--failed-kind" in argv:
        position = argv.index("--failed-kind")
        if position + 1 >= len(argv):
            raise ItemsError("--failed-kind needs a value")
        failed_kind = argv[position + 1]
    if not node_id:
        raise ItemsError("--node is required: the declarations are keyed by node id")
    return node_id, index, failed_reason, failed_kind


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = (
        "usage: python -m refactory_dsh.items "
        "{list|begin|end} --node <id> [--index <n>] [--failed <reason>] | reset < redo-plan.json"
    )
    if not argv or argv[0] not in ("list", "begin", "end", "reset"):
        print(usage, file=sys.stderr)
        return 2
    verb = argv[0]
    workdir = Path(os.environ.get(WORKDIR_ENV) or project_for_nodes())
    try:
        from refactory_dsh.run import install_workspace_packages

        install_workspace_packages()
        if verb == "reset":
            produced = command_reset(workdir, json.load(sys.stdin))
            print(json.dumps(produced, default=str))
            return 0
        node_id, index, failed_reason, failed_kind = _options(argv[1:])
        node = data_node(workdir, node_id)
        if verb == "list":
            produced: dict[str, Any] = command_list(node, workdir)
        elif verb == "begin":
            produced = command_begin(node, workdir, index)
        else:
            produced = command_end(
                node, workdir, index,
                failed_reason=failed_reason, failed_kind=failed_kind,
            )
    except AllItemsFailedError as exc:
        # The all-fail rule gets its own exit code so the spine can
        # distinguish it from other end failures (Ask 3).
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return ALL_FAIL_EXIT_CODE
    except (ItemsError, TaskError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # a failing item loop must fail loudly, not traceback
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(produced, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
