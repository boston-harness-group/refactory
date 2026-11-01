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

import json
import os
import sys
from pathlib import Path
from typing import Any

from refactory_dsh.paths import project_root, project_for_nodes
from refactory_dsh.nn import Node
from refactory_dsh.nn.executor import CURRENT_ITEM, ITEMS_OVERRIDE, load_items
from refactory_dsh.task import Task, TaskError, instance_of, load_task

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


class ItemsError(RuntimeError):
    """A data node cannot be iterated: named, because the node id alone is not a reason."""


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
        facts[fact] = list(previous) + scores if isinstance(previous, list) else list(scores)

    # Under the facts lock, as every node's write is: a data pass can end beside a branch.
    edit_facts(workdir, add)


def command_list(node: Node, workdir: Path) -> dict[str, Any]:
    """How many items the node iterates, and what they are called."""
    items = items_of(node, workdir)
    return {
        "count": len(items),
        "ids": [str(item.get("id") or index) for index, item in enumerate(items)],
    }


def command_begin(node: Node, workdir: Path, index: int) -> dict[str, Any]:
    """Materialise one item and prepare the workspace for its pass."""
    items = items_of(node, workdir)
    if index < 0 or index >= len(items):
        raise ItemsError(
            f"item {index} is out of range: node {node.id!r} selected {len(items)} item(s)"
        )
    item = items[index]
    (workdir / CURRENT_ITEM).write_text(json.dumps(item, indent=2, default=str))
    # Clear the exchange sidecar for this item's pass.
    sidecar = workdir / EXCHANGES_SIDECAR
    if sidecar.exists():
        sidecar.unlink(missing_ok=True)
    task = _task_of(node, workdir)
    if task is not None:
        task.setup(instance_of(item), workdir)
    return item


def make_item_record(
    item_id: str,
    *,
    passed: bool,
    score: float,
    exchanges: list[dict[str, Any]] | None = None,
    prediction: str | None = None,
    label: str | None = None,
    label_text: str | None = None,
    fail_reason: str | None = None,
    error_kind: str | None = None,
    error_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a per-item record for one evaluated item.

    One record per item, same shape on both runtimes.  Extends F4's existing
    error_kind convention (exception class name or ``"timeout"``).  The
    ``exchanges`` list carries one ``{node, prompt, response}`` dict per agent
    call in the item's pass, so a multi-node pass records every model call.
    """
    return {
        "id": item_id,
        "passed": passed,
        "score": score,
        "exchanges": list(exchanges or []),
        "prediction": prediction,
        "label": label,
        "label_text": label_text,
        "fail_reason": fail_reason,
        "error_kind": error_kind,
        "error_text": error_text,
        "metadata": dict(metadata or {}),
    }


def command_end(node: Node, workdir: Path, index: int) -> dict[str, Any]:
    """The Task's verdict on one pass, folded into the run's facts."""
    items = items_of(node, workdir)
    if index < 0 or index >= len(items):
        raise ItemsError(
            f"item {index} is out of range: node {node.id!r} selected {len(items)} item(s)"
        )
    item = items[index]
    task = _task_of(node, workdir)
    if task is None:
        return {}
    instance = instance_of(item)
    verdict = task.verify(instance, workdir)
    if verdict is None:
        return {}
    item_id = str(item.get("id") or index)
    metadata = {k: v for k, v in item.items() if k not in ("id", "path")}
    # Read per-node exchanges from the sidecar (written by agent execution).
    exchanges = _read_and_clear_exchanges(workdir)
    record = make_item_record(
        item_id,
        passed=verdict.passed,
        score=verdict.score,
        exchanges=exchanges,
        prediction=verdict.details if verdict.passed else None,
        label=str(instance.metadata.get("label", "")) if instance.metadata.get("label") is not None else None,
        label_text=str(instance.metadata.get("label_text", "")) if instance.metadata.get("label_text") is not None else None,
        fail_reason=verdict.details if not verdict.passed and verdict.details else None,
        metadata=metadata,
    )
    if SCORES_FACT in node.produces:
        _write_facts(workdir, SCORES_FACT, [{"id": item_id, "passed": verdict.passed, "score": verdict.score}])
    return record


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



def _options(argv: list[str]) -> tuple[str, int]:
    """The ``--node`` id and ``--index`` one command line names."""
    node_id = ""
    index = 0
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
    if not node_id:
        raise ItemsError("--node is required: the declarations are keyed by node id")
    return node_id, index


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = (
        "usage: python -m refactory_dsh.items "
        "{list|begin|end} --node <id> [--index <n>] | reset < redo-plan.json"
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
        node_id, index = _options(argv[1:])
        node = data_node(workdir, node_id)
        if verb == "list":
            produced: dict[str, Any] = command_list(node, workdir)
        elif verb == "begin":
            produced = command_begin(node, workdir, index)
        else:
            produced = command_end(node, workdir, index)
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
