"""Synthetic SpreadsheetBench 2 workbooks for the tests, from JSON specs.

As `sb_support.make_workbook`, and a cell may also be ``{"f": <formula>, "v": <cached
value>}``: a formula with the value Excel (or LibreOffice) cached for it, which openpyxl
cannot write, set in the saved file's XML. ``fixtures/spreadsheetbench_v2_grader_parity.json``
was recorded by running SpreadsheetBench 2's own ``evaluation.py`` on workbooks built by
this function (`record_parity`); the specs and V2's results are both in it.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from sb_support import _value
from sb_support import make_workbook as _make_plain


def make_workbook(path: Path, spec: list[dict[str, Any]]) -> Path:
    plain, cached = [], []
    for sheet in spec:
        cells = {}
        for coordinate, value in sheet["cells"].items():
            if isinstance(value, dict) and "f" in value:
                cells[coordinate] = value["f"]
                cached.append((len(plain), coordinate, value["v"]))
            else:
                cells[coordinate] = value
        plain.append({"title": sheet["title"], "cells": cells})
    _make_plain(path, plain)
    if cached:
        _cache(path, cached)
    return path


def _cache(path: Path, cached: list[tuple[int, str, Any]]) -> None:
    """Give formula cells cached values in the saved file (openpyxl writes ``<v />``)."""
    with zipfile.ZipFile(path) as zin:
        members = {name: zin.read(name) for name in zin.namelist()}
    for index, coordinate, value in cached:
        name = f"xl/worksheets/sheet{index + 1}.xml"
        xml = members[name].decode()
        if isinstance(value, str) and value.startswith("#"):
            kind, text = "e", value
        elif isinstance(value, bool):
            kind, text = "b", str(int(value))
        elif isinstance(value, (int, float)):
            kind, text = "n", repr(value)
        else:
            kind, text = "str", str(value)
        pattern = re.compile(rf'<c r="{coordinate}"( [^>]*)?><f>(.*?)</f><v ?/>(</v>)?</c>')
        xml, n = pattern.subn(lambda m: f'<c r="{coordinate}" t="{kind}"><f>{m.group(2)}</f>'
                                        f"<v>{text}</v></c>", xml)
        assert n == 1, (coordinate, xml[-300:])
        members[name] = xml.encode()
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    shutil.move(tmp, path)


def regression_case(wrong: int, mod_ok: bool = True, rows: int = 1000) -> dict[str, Any]:
    """``rows`` unchanged cells (``S!C1:C<rows>``) and one changed (``S!D1``), ``wrong`` of
    the unchanged ones broken in the output: V2's 99.8% rule, read over ``S!C1:D<rows>``
    (the empty D cells unchanged too)."""
    inp = {f"C{i}": i for i in range(1, rows + 1)}
    gold = {**inp, "D1": 5}
    out = {**gold, "D1": 5 if mod_ok else 6, **{f"C{i}": -i for i in range(1, wrong + 1)}}
    return {key: [{"title": "S", "cells": cells}]
            for key, cells in (("input", inp), ("golden", gold), ("output", out))}


def build_case(root: Path, case: dict[str, Any]) -> tuple[Path, Path, Path]:
    """The case's input, golden and output workbooks under ``root`` (the output absent
    when the case has none); a ``regression`` case is built by `regression_case`."""
    if "regression" in case:
        case = {**case, **regression_case(**case["regression"])}
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for role in ("input", "golden", "output"):
        path = root / f"{role}.xlsx"
        if case.get(role) is not None:
            make_workbook(path, case[role])
        paths.append(path)
    return paths[0], paths[1], paths[2]


def decode(value: Any) -> Any:
    return _value(value)


def record_parity(evaluation_py: str, specs: str, out: str, work: str) -> None:
    """Run V2's own ``evaluation.py`` on every case and value pair of ``specs`` and write
    the fixture: ``process_single_item``'s result for each case (on a dataset laid out as
    V2 reads one), and ``compare_cell_value`` / ``compare_cell_formula`` of each pair."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sb2_evaluation", evaluation_py)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    data = json.loads(Path(specs).read_text())
    results = {}
    for case in data["cases"]:
        root = Path(work) / case["name"]
        _, _, output = build_case(root, case)
        outputs = root / "outputs"
        outputs.mkdir(exist_ok=True)
        if output.exists():
            shutil.copyfile(output, outputs / f"{case['name']}_output.xlsx")
        item = {"id": case["name"], "spreadsheet_path": "input.xlsx",
                "golden_response_path": "golden.xlsx", "answer_position": case["answer_position"]}
        result, missing = ev.process_single_item(item, str(root), str(outputs), {}, "Template")
        results[case["name"]] = {"accuracy": result["accuracy"],
                                 "regression_accuracy": result["regression_accuracy"],
                                 "modification_accuracy": result["modification_accuracy"],
                                 "error_message": result["error_message"], "missing": missing}
        full = ev.compare_workbooks_with_regression(
            str(root / "input.xlsx"), str(root / "golden.xlsx"), str(outputs / f"{case['name']}_output.xlsx"),
            case["answer_position"])
        results[case["name"]].update({"regression": full[4], "modification": full[5]})
    pairs = [{"a": a, "b": b, "value": ev.compare_cell_value(decode(a), decode(b)),
              "transform_a": repr(ev.transform_value(decode(a)))} for a, b in data["pairs"]]
    data["v2"] = {"cases": results, "pairs": pairs,
                  "evaluation_py_sha256": __import__("hashlib").sha256(Path(evaluation_py).read_bytes()).hexdigest()}
    Path(out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
