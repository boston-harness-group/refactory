"""Synthetic SpreadsheetBench workbooks for the tests, from JSON specs.

A spec is ``[{"title": <sheet>, "cells": {<coordinate>: <value>}}, ...]``; a value is a
JSON scalar, or ``{"datetime": <iso>}``, ``{"time": <iso>}``. The parity fixture's
expected values (``fixtures/spreadsheetbench_parity.json``) were produced by SkillOpt's
own code on workbooks built by this function.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


def _value(value: Any) -> Any:
    if isinstance(value, dict) and "datetime" in value:
        return datetime.datetime.fromisoformat(value["datetime"])
    if isinstance(value, dict) and "time" in value:
        return datetime.time.fromisoformat(value["time"])
    return value


def make_workbook(path: Path, spec: list[dict[str, Any]]) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for sheet in spec:
        ws = wb.create_sheet(sheet["title"])
        for coordinate, value in sheet["cells"].items():
            ws[coordinate] = _value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def make_dataset(root: Path, items: list[dict[str, Any]], parts: dict[str, list[str]]) -> Path:
    """A dataset directory (``dataset.json`` and ``spreadsheet/<id>/``) and a manifest
    for it, in the release's layout. Each item carries ``init`` and (unless it has
    none) ``gold`` workbook specs; ``gold_name`` overrides the golden file's name.
    Returns the manifest path."""
    import hashlib

    data = root / "sb400"
    rows = []
    for item in items:
        item = dict(item)
        init, gold = item.pop("init"), item.pop("gold", None)
        gold_name = item.pop("gold_name", f"1_{item['id']}_golden.xlsx")
        folder = data / "spreadsheet" / item["id"]
        make_workbook(folder / f"1_{item['id']}_init.xlsx", init)
        if gold is not None:
            make_workbook(folder / gold_name, gold)
        rows.append({"spreadsheet_path": f"spreadsheet/{item['id']}", **item})
    (data / "dataset.json").write_text(json.dumps(rows))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "source": "synthetic", "revision": "test", "archive": "none",
        "dataset_json_sha256": hashlib.sha256((data / "dataset.json").read_bytes()).hexdigest(),
        "parts": parts}))
    return manifest
