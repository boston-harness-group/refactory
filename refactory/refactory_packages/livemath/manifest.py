"""Parts manifest builder for the LiveMathematicianBench id split.

Reads train/val/test id lists from split_dir (the same layout as
~/skillopt/data/livemathematicianbench_id_split) and writes a parts.json
manifest in the format the outer loop's ``parts_manifest`` setting takes (#558).
"""
from __future__ import annotations

import json
from pathlib import Path

from refactory_dsh.task import make_manifest


def build_parts_manifest(split_dir: str | Path, output_dir: str | Path) -> Path:
    """Build a parts.json manifest from a split directory.

    ``split_dir`` contains ``train/items.json``, ``val/items.json`` and
    ``test/items.json``, each a JSON array of objects with an ``"id"`` field
    (matching the structure at ~/skillopt/data/livemathematicianbench_id_split).

    Optionally validates counts against ``split_manifest.json`` in ``split_dir``
    if it exists.
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    train_ids = _load_ids(split_dir / "train" / "items.json")
    val_ids = _load_ids(split_dir / "val" / "items.json")
    test_ids = _load_ids(split_dir / "test" / "items.json")

    # Optionally validate against split_manifest.json
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        counts = manifest.get("counts", {})
        for name, ids, expected in [
            ("train", train_ids, counts.get("train")),
            ("val", val_ids, counts.get("val")),
            ("test", test_ids, counts.get("test")),
        ]:
            if expected is not None and len(ids) != expected:
                raise ValueError(
                    f"split {name} has {len(ids)} ids but split_manifest.json "
                    f"says {expected}"
                )

    return make_manifest(
        output_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )


def _load_ids(path: Path) -> list[str]:
    """Load item ids from a JSON array file."""
    data = json.loads(path.read_text())
    return [str(item["id"]) for item in data]
