"""Every evaluated item's full trace, written as files in the run folder.

After each evaluation the loop hands the evaluation's per-item records here
(`Loop._measure`, the one place every evaluation passes through on both runtimes:
the walker's records and the spine's ``item_records.json`` alike), and each item's
record is written out as::

    traces/<candidate>/<part>-<occasion>[-seed<seed>][-eval<n>][-repeat<n>]/<item>/
        NN-<node>.json   one per model call, in run order: the exchange as recorded
        steps.json       the op and code-gate visits, when the item has any
        item.json        the record without its exchanges and steps (the original id
                         in "id"), written last: an item folder with it is complete

``-eval<n>`` numbers a second and later evaluation the journal records in the same
evaluation folder (two train rollouts of one candidate at one seed); an attempt the
journal does not record (a retry, a resume running an interrupted evaluation again)
rewrites its own folder: each file atomically, the call files the new trace no longer
has removed.

Write-only for now: nothing in the loop, the optimizer, its analysts, the memory pass,
the reports or a resume reads these files, and a test holds that (no string literal of
the package or the TypeScript sources names the folder or these file names outside a
listed set of other uses). The run journal stays the record; this is a best-effort
archive for people and a later trace-reading analyst. An item that cannot be written is
skipped, the others are written, and the evaluation logs one warning line.
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


def occasion_name(part: str, occasion: str, seed: int | None, repeat: int = 1,
                  number: int = 1) -> str:
    """The folder name of one evaluation: its part, its occasion (without a leading
    repeat of the part's name), the rollout seed and, past the first, the evaluation's
    number in its folder and the repeat."""
    occasion = str(occasion or "")
    if occasion.startswith(f"{part}_"):
        occasion = occasion[len(part) + 1:]
    name = "-".join(bit for bit in (str(part), occasion) if bit)
    if seed is not None:
        name += f"-seed{seed}"
    if number > 1:
        name += f"-eval{number}"
    if repeat > 1:
        name += f"-repeat{repeat}"
    return safe_name(name)


def _dump(path: Path, value: Any) -> None:
    """Write ``value`` as indented JSON to ``path`` atomically: a temp file beside it,
    then a rename, so a reader never sees half a file and a rewrite never corrupts one."""
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        # A lone surrogate in a reply is written escaped rather than failing the file.
        with os.fdopen(handle, "w", encoding="utf-8", errors="backslashreplace") as out:
            json.dump(value, out, indent=1, default=str, ensure_ascii=False)
            out.write("\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write_item(folder: Path, record: Mapping[str, Any]) -> None:
    """One item's trace in ``folder``: a file per model call, ``steps.json``, and
    ``item.json`` last; an earlier trace's ``item.json`` is removed first, and the call
    and steps files it left that this one does not have afterwards."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "item.json").unlink(missing_ok=True)
    written = {"item.json"}
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
    _dump(folder / "item.json", {k: v for k, v in record.items() if k not in _SPLIT})


def clear(root: Path) -> None:
    """Remove ``root``'s traces: a fresh run in a directory starts them over, as it
    starts the journal over, since the journal numbers them. Never raises."""
    import shutil

    shutil.rmtree(Path(root) / TRACES_DIR, ignore_errors=True)


def write_evaluation(root: Path, candidate: str, part: str, occasion: str,
                     seed: int | None, records: Iterable[Mapping[str, Any]], *,
                     number: int = 1) -> None:
    """Write every record of ``candidate``'s evaluation number ``number`` in its
    folder under ``root/traces``.

    Never raises: an item that cannot be written is skipped and the others are written,
    and one warning line says how many failed, since the run's own record is the
    journal, not these files."""
    failed: list[str] = []
    first: BaseException | None = None
    try:
        base = Path(root) / TRACES_DIR / safe_name(candidate)
        for record in records:
            if not isinstance(record, Mapping):
                continue
            try:
                repeat = record.get("repeat")
                folder = base / occasion_name(part, occasion, seed,
                                              repeat if isinstance(repeat, int) else 1, number)
                write_item(folder / safe_name(record.get("id")), record)
            except Exception as exc:  # noqa: BLE001 - an archive, never the run's outcome
                failed.append(str(record.get("id")))
                first = first or exc
    except Exception as exc:  # noqa: BLE001 - records that cannot even be iterated
        failed.append("?")
        first = first or exc
    if failed:
        _log.warning("refactory: could not write %d of %s's %s item traces (first %s): %s: %s",
                     len(failed), candidate, part, failed[0], type(first).__name__, first)


__all__ = ["TRACES_DIR", "clear", "occasion_name", "safe_name", "write_evaluation", "write_item"]
