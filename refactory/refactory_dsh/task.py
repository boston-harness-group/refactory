"""The Task protocol: what a dataset mode's items are, and how one is judged.

A dataset mode iterates its subgraph once per item. Where the items come from, and
whether a pass was any good, is the Task's business rather than the graph's — the
same split remote-factory's `factory/task.py` makes, so a mode bound to a benchmark
or a dataset carries that binding as a declaration instead of as instructions in a
prompt.

A Task is three hooks, all optional:

- ``instances()`` — the items. The default is one unnamed instance, which is what a
  mode that only needs to run once wants.
- ``setup(instance, workspace)`` — prepare the workspace before the pass.
- ``verify(instance, workspace)`` — judge the pass, or return ``None`` to say the
  Task has no opinion and the mode's own score stands.

There is deliberately no hook that writes the prompt. An item is data — it reaches the
pass as ``current_item.json`` — and what the agent for an item is asked is the mode's
own business, stated in its agent node's prompt template. Two places that both decide
the ask is one place too many.

Resolution is by ``module:Class`` or by a name registered here, and a ref that does
not resolve is refused by name: a mode that names a Task nobody can load must not
silently run its items unverified.

A run directory may also say which *part* of the items it evaluates. The outer loop
writes that choice beside the run, and the loader both runtimes use reads it, so a
search is examined on the items it was tuned against and the winner can be examined
on the items no candidate saw. The choice is a file rather than a knob: a search
that could set it would take its own exam.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class TaskError(ValueError):
    """A Task reference does not resolve, or a Task misbehaves."""


@dataclass(frozen=True)
class TaskInstance:
    """One item a Task says the mode has to process."""

    id: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifyResult:
    """What one pass was worth."""

    passed: bool
    score: float
    details: str = ""
    prediction: str = ""


class Task:
    """The three hooks a dataset binding implements."""

    name = ""
    #: Item fields only the grader may see: the answer, and anything that states it.
    #: ``verify`` gets the whole item (its instance), but the per-item file the
    #: graph's nodes read (``current_item.json``) never carries these, on either
    #: runtime (`node_item`). A node an optimizer adds can read that file, so a Task
    #: whose items hold their own answer has to say so here. Default: none.
    grader_fields: tuple[str, ...] = ()
    #: Grader-only fields a run may still show the nodes, each behind the knob that
    #: turns it on (``{"theorem": "use_theorem"}``). The field reaches
    #: ``current_item.json`` only in a run whose knob is true; a mode that declares
    #: such a knob freezes it, so no search can turn it on.
    revealed_by: dict[str, str] = {}

    def instances(
        self, *, part: str | None = None, seed: int | None = None,
    ) -> list[TaskInstance]:
        """The items to process. One unnamed instance unless the Task says otherwise.

        ``part`` and ``seed`` are optional: a Task that depends on neither
        ignores them (backward compatible).  A Task whose items vary by seed
        overrides this and uses them.
        """
        return [TaskInstance(id="default")]

    def setup(
        self, instance: TaskInstance, workspace: Path,
        *, part: str | None = None, seed: int | None = None,
    ) -> None:
        """Prepare the workspace for one item. Nothing to do by default."""

    def verify(
        self, instance: TaskInstance, workspace: Path,
        *, part: str | None = None, seed: int | None = None,
    ) -> VerifyResult | None:
        """Judge one pass, or ``None`` when the Task has no opinion."""
        return None


#: The Tasks a bare name can resolve to.
TASKS: dict[str, type[Task]] = {}


def register_task(name: str, cls: type[Task]) -> None:
    """Make a Task resolvable by name, and return nothing."""
    TASKS[name] = cls


def load_task(ref: str, project: Path | None = None) -> Task:
    """Resolve a Task reference to an instance.

    ``module:Class`` imports it; a bare name is looked up in :data:`TASKS`. Anything
    else is refused with the ref in the message, because a mode bound to a Task that
    cannot be loaded would run its items with nothing judging them.
    """
    ref = (ref or "").strip()
    if not ref:
        raise TaskError("an empty Task reference names no Task")
    if ":" in ref:
        module_name, _, attribute = ref.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise TaskError(f"cannot import {module_name!r} for Task {ref!r}: {exc}") from exc
        cls = getattr(module, attribute, None)
        if cls is None:
            raise TaskError(f"{module_name!r} has no attribute {attribute!r} (Task {ref!r})")
        if not isinstance(cls, type) or not issubclass(cls, Task):
            raise TaskError(f"{ref!r} is not a Task subclass")
        return cls()
    if ref not in TASKS:
        known = ", ".join(sorted(TASKS)) or "none"
        raise TaskError(f"no Task named {ref!r} is registered; registered: {known}")
    return TASKS[ref]()


def node_item(item: dict[str, Any], task: Task | None, knobs: dict[str, Any] | None) -> dict[str, Any]:
    """The item as the graph's nodes may read it: without the Task's grader-only fields.

    A grader-only field named in ``revealed_by`` stays when the run's knob for it is
    true. Both runtimes write ``current_item.json`` through here; the grader keeps the
    whole item (``instance_of``).
    """
    hidden = set(getattr(task, "grader_fields", ()) or ())
    if not hidden:
        return item
    revealed = getattr(task, "revealed_by", None) or {}
    knobs = knobs or {}
    shown = {name for name in hidden if name in revealed and bool(knobs.get(revealed[name]))}
    return {key: value for key, value in item.items() if key not in hidden - shown}


def instance_of(item: dict[str, Any]) -> TaskInstance:
    """The Task instance one loaded item describes."""
    metadata = {
        key: value for key, value in item.items() if key not in ("id", "path")
    }
    return TaskInstance(
        id=str(item.get("id") or ""),
        path=str(item["path"]) if item.get("path") else None,
        metadata=metadata,
    )


# ── the part of the items a run directory evaluates ────────────────────────

#: The file in a run directory that names the part of the items it evaluates.
SUBSET = "subset.json"

#: The parts an item selection can name.
TRAIN = "train"
HOLDOUT = "holdout"
VAL = "val"
TEST = "test"

#: The holdout fraction used when a run directory names a part but no fraction.
DEFAULT_HOLDOUT = 0.3

#: An explicit id-list manifest for three-way partitioning.
MANIFEST = "parts.json"


def _item_key(item: dict[str, Any], index: int) -> str:
    """The id an item is partitioned by, falling back to its position."""
    return str(item.get("id") or index)


def partition(
    items: Sequence[dict[str, Any]], holdout: float
) -> dict[str, list[dict[str, Any]]]:
    """Split items into a train part and a holdout part, by sorted instance id.

    The partition is deterministic, so both runtimes and the audit agree about which
    items are which. The ids are sorted, the last ``holdout`` fraction of them is
    held out, and at least one item stays in train whenever there are two. Fewer than
    two items cannot be split at all, and the holdout part is then empty rather than
    an error: one instance is a fact about the Task, not a reason to fail a run.
    """
    fraction = min(max(float(holdout), 0.0), 1.0)
    order = sorted(
        range(len(items)), key=lambda index: (_item_key(items[index], index), index)
    )
    if len(order) < 2 or fraction <= 0.0:
        return {TRAIN: list(items), HOLDOUT: []}
    count = min(len(order) - 1, max(1, int(round(len(order) * fraction))))
    held = set(order[-count:])
    return {
        TRAIN: [item for index, item in enumerate(items) if index not in held],
        HOLDOUT: [item for index, item in enumerate(items) if index in held],
    }


def counts(items: Sequence[dict[str, Any]], holdout: float) -> dict[str, int]:
    """How many items each part holds, without materialising either part."""
    parts = partition(items, holdout)
    return {TRAIN: len(parts[TRAIN]), HOLDOUT: len(parts[HOLDOUT])}


def _subset_file(workdir: Path) -> Path:
    """Where a run directory records the part of the items it evaluates."""
    return Path(workdir) / SUBSET


def make_manifest(
    workdir: Path,
    *,
    train_ids: list[str],
    val_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
) -> Path:
    """Write an explicit id-list manifest for three-way partitioning."""
    manifest: dict[str, list[str]] = {"train": list(train_ids)}
    if val_ids is not None:
        manifest["val"] = list(val_ids)
    if test_ids is not None:
        manifest["test"] = list(test_ids)
    path = Path(workdir) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def split_to_manifest(
    workdir: Path,
    items: Sequence[dict[str, Any]],
    *,
    holdout: float = DEFAULT_HOLDOUT,
    val_fraction: float = 0.0,
) -> Path:
    """Partition items by fractions and write the resulting id lists as a manifest."""
    parts = partition(items, holdout)
    train_items = parts[TRAIN]
    test_items = parts[HOLDOUT]
    train_ids = [str(item.get("id") or i) for i, item in enumerate(items) if item in train_items]
    test_ids = [str(item.get("id") or i) for i, item in enumerate(items) if item in test_items]
    val_ids: list[str] | None = None
    if val_fraction > 0 and len(train_ids) > 1:
        split_point = max(1, int(round(len(train_ids) * (1 - val_fraction))))
        val_ids = train_ids[split_point:]
        train_ids = train_ids[:split_point]
    return make_manifest(workdir, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids or None)


def manifest_part_ids(manifest: Mapping[str, Any], part: str) -> list[str] | None:
    """The ids a parts manifest gives ``part``, or ``None`` when it gives it none.

    The holdout audit's part is the manifest's ``val`` part (empty when it has
    none): those items are not train, so the search never scored them, and they
    are not test, which nothing but a strategy's own final evaluation may touch.
    Splitting off a fraction instead would draw the holdout from every item the
    Task has, test items included.
    """
    if part in manifest:
        return [str(x) for x in manifest[part]]
    if part == HOLDOUT:
        return [str(x) for x in manifest.get(VAL, [])]
    return None


def write_subset(workdir: Path, part: str, holdout: float, *, seed: int | None = None) -> Path:
    """Record which part of a Task's items a run directory evaluates.

    The outer loop writes this before it evaluates a candidate and the item loader
    reads it, so the part travels with the run directory rather than with the search.
    """
    path = _subset_file(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"part": str(part), "holdout": float(holdout)}
    if seed is not None:
        data["seed"] = seed
    path.write_text(json.dumps(data, indent=2))
    return path


def read_subset(workdir: Path) -> dict[str, Any]:
    """The part a run directory names, or ``{}`` when it names none."""
    path = _subset_file(workdir)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def subset_part_seed(workdir: Path) -> tuple[str | None, int | None]:
    """The part and seed ``subset.json`` names, each ``None`` when it names none."""
    record = read_subset(workdir)
    part = record.get("part") or None
    seed = record.get("seed")
    return (str(part) if part else None), (int(seed) if seed is not None else None)


def part_seed_kwargs(
    method: Any, *, part: str | None, seed: int | None,
) -> dict[str, Any]:
    """The ``part``/``seed`` keywords a Task hook accepts, decided by its signature.

    A Task written before ``part`` and ``seed`` existed takes neither, so they are
    passed only to a hook that names them or takes ``**kwargs``. Deciding from the
    signature rather than by calling and catching ``TypeError`` means a TypeError
    raised inside the hook is that item's failure, never a silent second call.
    """
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"part": part, "seed": seed}
    accepted = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {
        name: value
        for name, value in (("part", part), ("seed", seed))
        if name in params and params[name].kind in accepted
    }


def select_subset(
    items: Sequence[dict[str, Any]], workdir: Path
) -> list[dict[str, Any]]:
    """The part of ``items`` a run directory names, or all of them when it names none.

    Called by the one item loader both runtimes use, so the walker and DSH iterate
    the same items of the same Task. When a ``parts.json`` manifest exists, item
    ids are looked up there. When ``seed`` is present in subset.json, the returned
    items are shuffled within the part using ``random.Random(seed)``.
    """
    import random as _random

    record = read_subset(workdir)
    part = str(record.get("part") or "")

    # Try manifest first
    manifest_path = Path(workdir) / MANIFEST
    if manifest_path.exists() and part:
        try:
            manifest = json.loads(manifest_path.read_text())
            ids = manifest_part_ids(manifest, part) if isinstance(manifest, dict) else None
            if ids is not None:
                allowed_ids = set(ids)
                selected = [
                    item for i, item in enumerate(items)
                    if str(item.get("id") or i) in allowed_ids
                ]
                seed = record.get("seed")
                if seed is not None:
                    selected = list(selected)
                    _random.Random(seed).shuffle(selected)
                return selected
        except (OSError, json.JSONDecodeError):
            pass

    if part not in (TRAIN, HOLDOUT, VAL, TEST):
        return list(items)
    if part in (VAL, TEST):
        raise TaskError(
            f"part {part!r} requires a parts manifest (parts.json); "
            f"no manifest was found in {workdir}"
        )
    effective = part
    try:
        fraction = float(record.get("holdout", DEFAULT_HOLDOUT))
    except (TypeError, ValueError):
        fraction = DEFAULT_HOLDOUT
    selected = partition(items, fraction)[effective]
    seed = record.get("seed")
    if seed is not None:
        selected = list(selected)
        _random.Random(seed).shuffle(selected)
    return selected
