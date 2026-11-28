"""Item loading and sampling for data nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..graph import Node
from .failures import ExecutionError

if TYPE_CHECKING:
    from refactory_dsh.task import Task

#: The file a data node's current item is written to, for the pass that reads it.
CURRENT_ITEM = "current_item.json"

#: Where a run says which items a data node should read, when they are not the ones
#: the mode's own graph declares. A harness for a class of work is built once and fed
#: many times, so the source is run configuration rather than graph structure.
ITEMS_OVERRIDE = "items.json"

#: Where a smoke's sample names the items each Task-bound data node runs
#: (`sandbox.smoke_sample`), relative to the run directory: ``{node id: [item ids]}``.
SMOKE_SAMPLE = ".refactory/smoke-sample.json"


def load_items(node: Node, workdir: Path) -> list[dict[str, Any]]:
    """The items a data node iterates, in the order it declares them.

    One loader, so a mode means the same thing on either runtime: the item is handed
    to the pass as JSON rather than as a template variable, because a pass is a
    subgraph of nodes and only the first of them sees a prompt.

    The part the run directory names is applied last, by
    :func:`refactory_dsh.task.select_subset`; both runtimes load through this
    function, so both iterate the same part of the same items.
    """
    from refactory_dsh.task import _item_key, select_subset

    every = full_items(node, workdir)
    items = select_subset(every, workdir)
    # A smoke's sample (`sandbox.smoke_sample`) names, per Task-bound node, the items it
    # runs; any other run iterates the whole selected part, as it always has. An item is
    # named as the sample and the selection name it: by id, else by its position among
    # all the node's items (`task._item_key`), never its position in the selection.
    if node.task_ref:
        sample = smoke_sample_ids(workdir, node.id)
        if sample is not None:
            keys = {id(item): _item_key(item, index) for index, item in enumerate(every)}
            items = [item for item in items if keys.get(id(item)) in sample]
    return items


def smoke_sample_ids(workdir: Path, node_id: str) -> set[str] | None:
    """The item ids a smoke's sample gives ``node_id``, or None outside a smoke."""
    try:
        sample = json.loads((Path(workdir) / SMOKE_SAMPLE).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ids = sample.get(node_id) if isinstance(sample, dict) else None
    return {str(item) for item in ids} if isinstance(ids, list) else None


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
