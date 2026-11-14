"""Every evaluated item's full trace, written as files in the run folder.

After each evaluation the loop hands the evaluation's per-item records here
(`Loop._measure`, the one place every evaluation passes through on both runtimes:
the walker's records and the spine's ``item_records.json`` alike), and each item's
record is written out as::

    traces/<candidate>/<part>-<occasion>[-seed<seed>][-repeat<n>]/<item>/
        item.json        the record without its exchanges and steps (the original id in "id")
        NN-<node>.json   one per model call, in run order: the exchange as recorded
        steps.json       the op and code-gate visits, when the item has any

Write-only for now: nothing in the loop, the optimizer, its analysts, the memory pass,
the reports or a resume reads these files, and a test holds that (only this module
names `TRACES_DIR`). The run journal stays the record; this is a best-effort archive
for people and a later trace-reading analyst. A write that fails logs one warning
line and the run carries on. A repeated evaluation of the same candidate on the
same occasion (a resume's re-score) overwrites each file atomically and removes the
call files the new trace no longer has.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from refactory_dsh.items import sanitize_item_id

#: The folder under the run root that holds the traces. Named only in this module.
TRACES_DIR = "traces"

#: The record fields kept out of ``item.json``: each has files of its own.
_SPLIT = ("exchanges", "steps")

#: The files this module writes in an item folder; only these are ever removed.
_OWN_FILE = re.compile(r"^(?:\d+-.+\.json|steps\.json|item\.json)$")

_log = logging.getLogger(__name__)


def safe_name(name: Any) -> str:
    """``name`` as one folder or file name: `items.sanitize_item_id` (the name pass
    folders use; a changed name gets a digest of the original, so two ids never
    share a folder), with the names a file system reserves (``""``, ``.``, ``..``)
    made safe the same way."""
    text = str(name)
    safe = sanitize_item_id(text)
    if safe in ("", ".", ".."):
        safe = f"_{hashlib.sha256(text.encode()).hexdigest()[:6]}"
    return safe


def occasion_name(part: str, occasion: str, seed: int | None, repeat: int = 1) -> str:
    """The folder name of one evaluation: its part, its occasion (without a leading
    repeat of the part's name), the rollout seed and, past the first, the repeat."""
    occasion = str(occasion or "")
    if occasion.startswith(f"{part}_"):
        occasion = occasion[len(part) + 1:]
    name = "-".join(bit for bit in (str(part), occasion) if bit)
    if seed is not None:
        name += f"-seed{seed}"
    if repeat > 1:
        name += f"-repeat{repeat}"
    return safe_name(name)


def _dump(path: Path, value: Any) -> None:
    """Write ``value`` as indented JSON to ``path`` atomically: a temp file beside it,
    then a rename, so a reader never sees half a file and a rewrite never corrupts one."""
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(value, out, indent=1, default=str, ensure_ascii=False)
            out.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write_item(folder: Path, record: Mapping[str, Any]) -> None:
    """One item's trace in ``folder``: ``item.json``, a file per model call and
    ``steps.json``; the call and steps files an earlier trace left that this one
    does not have are removed."""
    folder.mkdir(parents=True, exist_ok=True)
    written = {"item.json"}
    _dump(folder / "item.json", {k: v for k, v in record.items() if k not in _SPLIT})
    calls = [e for e in record.get("exchanges") or () if isinstance(e, Mapping)]
    width = max(2, len(str(len(calls))))
    for number, exchange in enumerate(calls, start=1):
        name = f"{number:0{width}d}-{safe_name(exchange.get('node') or 'call')}.json"
        _dump(folder / name, dict(exchange))
        written.add(name)
    steps = record.get("steps")
    if steps:
        _dump(folder / "steps.json", list(steps))
        written.add("steps.json")
    for path in folder.iterdir():
        if path.name not in written and _OWN_FILE.match(path.name):
            path.unlink(missing_ok=True)


def write_evaluation(root: Path, candidate: str, part: str, occasion: str,
                     seed: int | None, records: Iterable[Mapping[str, Any]]) -> None:
    """Write every record of one evaluation of ``candidate`` under ``root/traces``.

    Never raises: a failure logs one warning line and the rest is skipped, since the
    run's own record is the journal, not these files."""
    try:
        base = Path(root) / TRACES_DIR / safe_name(candidate)
        for record in records:
            if not isinstance(record, Mapping):
                continue
            repeat = record.get("repeat")
            folder = base / occasion_name(part, occasion, seed,
                                          repeat if isinstance(repeat, int) else 1)
            write_item(folder / safe_name(record.get("id")), record)
    except Exception as exc:  # noqa: BLE001 - an archive, never the run's outcome
        _log.warning("refactory: could not write %s's %s traces: %s: %s",
                     candidate, part, type(exc).__name__, exc)


__all__ = ["TRACES_DIR", "occasion_name", "safe_name", "write_evaluation", "write_item"]
