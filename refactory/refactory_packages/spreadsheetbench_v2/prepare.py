"""Prepare SpreadsheetBench 2 once: drop, rename, recalculate, freeze.

``python -m refactory_packages.spreadsheetbench_v2.prepare prepare SOURCE OUT`` reads the
extracted release (``SOURCE``: the ``spreadsheetbench-v2`` folder of
``spreadsheetbench-v2-09-10.zip``, holding ``Debugging/``, ``Financial_Model/``,
``Template/``), checks every file it uses against ``manifest.json`` (``SB2_MANIFEST``
names another), and writes ``OUT``:

- each kept task's input as ``<Category>/<group>/<id>/input.xlsx`` (so a Debugging input's
  name no longer names its planted error) and each golden once, under its group,
  both recalculated by LibreOffice as V2's ``open_spreadsheet.py`` does (`open_spreadsheet`)
  -- V2's README asks for this before any run, and the grader reads cached values;
- ``tasks.json``: the kept tasks with V2's instruction and answer position;
- ``prepared.json``: the sha256 of ``manifest.json``, ``tasks.json`` and every file
  written, and the LibreOffice version. The Task refuses data that does not match it.

It works one file at a time (one LibreOffice per file) and resumes: a file already
recalculated and recorded is not done again. A file LibreOffice cannot recalculate stops
it, naming the file.

``check OUT`` verifies the checksums; ``check --grades OUT`` also grades, with V2's grader,
each task's golden against itself (it must pass) and its input (it must fail), one task
at a time, and records any that do not in ``prepared.json`` (``sanity``).

``manifest SOURCE`` (maintainers) rebuilds ``manifest.json`` from the release: the drops,
the grouped split and the source checksums.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

MANIFEST = Path(__file__).resolve().with_name("manifest.json")
#: Another manifest in the same format (a smoke split, a test fixture); the shipped one by
#: default, as SB's ``SB_MANIFEST``.
MANIFEST_ENV = "SB2_MANIFEST"
#: What V2's grader file is, as the release ships it (``evaluation/evaluation.py``).
CATEGORIES = ("Debugging", "Financial_Model", "Template")
TASKS_FILE = "tasks.json"
PREPARED_FILE = "prepared.json"
PROGRESS_FILE = ".prepare-progress.json"

#: The grouped split: whole groups only, since a Debugging workbook's 10 tasks share one
#: golden (and each sibling's input holds the others' fixes), as an FM project's 5 do.
SPLIT_SEED = 20261123
#: Groups drawn per category, train then val (the rest is test), by `split_groups`.
SPLIT_COUNTS = {"Debugging": (2, 1), "Financial_Model": (5, 3)}
#: Template groups, chosen by the audit (each task has its own golden; train 25, val 15).
TEMPLATE_GROUPS = {"train": ["16_re_direct_cap", "14_re_dcf", "03_consolidation"],
                   "val": ["01_bond_accounting", "02_cash_sweep"]}

DROP_VISUALIZATION = ("Visualization needs V2's Windows Excel/WPS COM export and a VLM "
                      "checklist judge (run_visual_vlm_checklist_eval.py); not file-graded")
DROP_EMBEDDED = ("Debugging 'Embedded Hardcode': V2 grades it by formula text "
                 "(with_formula), so only the golden's exact formulas pass; dropped by the "
                 "owner as impossible")
DROP_COLOR = ("Debugging 'Inconsistent Color Coding': V2 grades font colours "
              "(with_font_color) against a colour convention the instruction never states; "
              "dropped by the owner as unspecified")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path() -> Path:
    override = os.environ.get(MANIFEST_ENV, "").strip()
    return Path(override).expanduser() if override else MANIFEST


def manifest_sha256() -> str:
    return sha256(manifest_path())


def load_manifest() -> dict[str, Any]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))


def group_of(row: dict[str, Any]) -> str:
    """The task's group: the folder under ``spreadsheet/`` (a Debugging workbook, an FM
    project, a Template)."""
    return str(row["spreadsheet_path"]).split("/")[1]


def _opaque(*parts: str) -> str:
    return hashlib.sha256(":".join((str(SPLIT_SEED), *parts)).encode()).hexdigest()[:12]


def item_id(category: str, source_id: str) -> str:
    """The item's id: opaque. A Debugging source id names its planted error (``01`` Double
    Counting, ``05`` Incorrect Average, ...), and the id reaches the agent's paths and both
    arms' analysts, so it is a seeded hash of the source; the source id stays in the
    manifest and the prepared ``tasks.json`` (grader side)."""
    return f"sb2-{_opaque(category, source_id)}"


def source_key(category: str, source_id: str) -> str:
    """The task's name in the release (manifest and grader side only)."""
    return f"{category}/{source_id}"


def drop_reason(category: str, row: dict[str, Any]) -> str:
    """Why a task is dropped, or "": Visualization, and the Debugging inputs V2 grades by
    formula ("Embedded") or font colour ("Color"), evaluation.py:431-436."""
    if category == "Visualization":
        return DROP_VISUALIZATION
    if category == "Debugging":
        name = os.path.basename(str(row.get("spreadsheet_path", "")))
        if "Embedded" in name:
            return DROP_EMBEDDED
        if "Color" in name:
            return DROP_COLOR
    return ""


def split_groups(groups: dict[str, list[str]], seed: int = SPLIT_SEED) -> dict[str, dict[str, list[str]]]:
    """Each category's groups as train/val/test: Debugging and Financial_Model drawn with
    ``random.Random(seed)`` (`SPLIT_COUNTS`, from the sorted groups), Template as the audit
    chose (`TEMPLATE_GROUPS`)."""
    rng = random.Random(seed)
    parts: dict[str, dict[str, list[str]]] = {}
    for category in CATEGORIES:
        names = sorted(groups.get(category, []))
        if category == "Template":
            train, val = list(TEMPLATE_GROUPS["train"]), list(TEMPLATE_GROUPS["val"])
            missing = [g for g in train + val if g not in names]
            if missing:
                raise ValueError(f"Template groups not in the release: {missing}")
        else:
            n_train, n_val = SPLIT_COUNTS[category]
            drawn = rng.sample(names, n_train + n_val)
            train, val = sorted(drawn[:n_train]), sorted(drawn[n_train:])
        test = [g for g in names if g not in train and g not in val]
        parts[category] = {"train": train, "val": val, "test": test}
    return parts


# ── the manifest (maintainers) ────────────────────────────────────────────────

def build_manifest(source: Path, *, archive_sha256: str = "", grader: Path | None = None) -> dict[str, Any]:
    """manifest.json from the extracted release at ``source``."""
    tasks, dropped = [], []
    groups: dict[str, set[str]] = {}
    datasets = {}
    for category in (*CATEGORIES, "Visualization"):
        path = source / category / "dataset.json"
        datasets[category] = sha256(path)
        for row in json.loads(path.read_text(encoding="utf-8")):
            reason = drop_reason(category, row)
            if reason:
                dropped.append({"category": category, "source_id": str(row["id"]),
                                "reason": reason})
                continue
            group = group_of(row)
            groups.setdefault(category, set()).add(group)
            folder = source / category
            tasks.append({
                "id": item_id(category, str(row["id"])), "category": category, "group": group,
                "source_id": str(row["id"]),
                "input": {"source": row["spreadsheet_path"],
                          "sha256": sha256(folder / row["spreadsheet_path"])},
                "golden": {"source": row["golden_response_path"],
                           "sha256": sha256(folder / row["golden_response_path"])},
            })
    split = split_groups({k: sorted(v) for k, v in groups.items()})
    parts: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for task in tasks:
        for part, names in split[task["category"]].items():
            if task["group"] in names:
                parts[part].append(task["id"])
    return {
        "benchmark": "SpreadsheetBench 2 (V2)",
        "source": "https://huggingface.co/datasets/KAKA22/SpreadsheetBench-v2",
        "archive": "spreadsheetbench-v2-09-10.zip",
        "archive_sha256": archive_sha256,
        "grader": "evaluation/evaluation.py",
        "grader_sha256": sha256(grader) if grader else "",
        "dataset_json_sha256": datasets,
        "split": {"seed": SPLIT_SEED, "unit": "group (Debugging workbook, FM project, Template)",
                  "method": ("Debugging and Financial_Model: random.Random(seed).sample of the "
                             "sorted groups, train then val (2+1 and 5+3 groups); Template: the "
                             "audit's choice; the rest test"),
                  "groups": split},
        "parts": parts,
        "dropped": dropped,
        # Tasks whose prepared golden fails against itself on the lab's full `check
        # --grades`: {"id", "reason"} each (`excluded_ids`).
        "excluded": [],
        "tasks": tasks,
    }


# ── prepare ───────────────────────────────────────────────────────────────────

class PrepareError(RuntimeError):
    pass


def prepared_paths(task: dict[str, Any]) -> tuple[str, str]:
    """Where the task's input and golden go in the prepared directory: named by the opaque
    id and a seeded hash of the golden's source, never by a source id or file name."""
    return (f"items/{task['id']}/input.xlsx",
            f"goldens/{_opaque(task['category'], task['golden']['source'])}.xlsx")


def _source_rows(source: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for category in CATEGORIES:
        path = source / category / "dataset.json"
        if not path.is_file():
            raise PrepareError(f"{path} not found: SOURCE is the extracted spreadsheetbench-v2 folder")
        if sha256(path) != manifest["dataset_json_sha256"][category]:
            raise PrepareError(f"{path} is not the release's (sha256 differs from manifest.json)")
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows[source_key(category, str(row["id"]))] = row
    return rows


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def prepare(source: Path, out: Path, *, recalculate=None) -> dict[str, Any]:
    """Write the prepared directory ``out`` from the release at ``source``; returns the
    ``prepared.json`` record. ``recalculate(src, dst) -> (ok, why)`` is
    `open_spreadsheet.recalculate` unless a test passes another."""
    from . import open_spreadsheet

    manifest = load_manifest()
    rows = _source_rows(Path(source), manifest)
    recalculate = recalculate or open_spreadsheet.recalculate
    if recalculate is open_spreadsheet.recalculate and not open_spreadsheet.available():
        raise PrepareError("LibreOffice not found: install it or set SB_SOFFICE")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / PROGRESS_FILE
    # Resume from an earlier prepare: its record, then an interrupted run's progress.
    progress: dict[str, str] = {}
    if (out / PREPARED_FILE).exists():
        progress.update(json.loads((out / PREPARED_FILE).read_text()).get("files", {}))
    if progress_path.exists():
        progress.update(json.loads(progress_path.read_text()))
    files: dict[str, str] = {}
    entries = []
    jobs: dict[str, tuple[Path, str]] = {}
    for task in manifest["tasks"]:
        row = rows.get(source_key(task["category"], task["source_id"]))
        if row is None:
            raise PrepareError(f"{source_key(task['category'], task['source_id'])} is not in "
                               "the release's dataset.json")
        input_rel, golden_rel = prepared_paths(task)
        folder = Path(source) / task["category"]
        for rel, ref in ((input_rel, task["input"]), (golden_rel, task["golden"])):
            jobs.setdefault(rel, (folder / ref["source"], ref["sha256"]))
        entries.append({"id": task["id"], "category": task["category"], "group": task["group"],
                        "source_id": task["source_id"], "instruction": row["instruction"],
                        "answer_position": row["answer_position"],
                        "input": input_rel, "golden": golden_rel})
    for n, (rel, (src, expected)) in enumerate(sorted(jobs.items()), 1):
        dst = out / rel
        if rel in progress and dst.exists() and sha256(dst) == progress[rel]:
            files[rel] = progress[rel]
            continue
        if sha256(src) != expected:
            raise PrepareError(f"{src} is not the release's file (sha256 differs from manifest.json)")
        _log(f"[{n}/{len(jobs)}] recalculating {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            ok, why = recalculate(str(src), str(dst))
        except open_spreadsheet.RecalcError as exc:
            ok, why = False, str(exc)
        if not ok:
            raise PrepareError(f"LibreOffice could not recalculate {src}: {why}")
        files[rel] = sha256(dst)
        progress[rel] = files[rel]
        progress_path.write_text(json.dumps(progress, indent=1, sort_keys=True))
    tasks_path = out / TASKS_FILE
    tasks_path.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    record = {"manifest_sha256": manifest_sha256(), "tasks_sha256": sha256(tasks_path),
              "libreoffice": open_spreadsheet.version(),
              "prepared_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "files": dict(sorted(files.items()))}
    (out / PREPARED_FILE).write_text(json.dumps(record, indent=1) + "\n")
    progress_path.unlink(missing_ok=True)
    return record


def check(out: Path) -> dict[str, Any]:
    """The prepared directory's record, once every checksum in it matches; raises
    `PrepareError` naming what does not."""
    out = Path(out)
    record_path = out / PREPARED_FILE
    if not record_path.is_file():
        raise PrepareError(f"{out} is not prepared: no {PREPARED_FILE} (run "
                           "`python -m refactory_packages.spreadsheetbench_v2.prepare prepare SOURCE OUT`)")
    record = json.loads(record_path.read_text())
    if record.get("manifest_sha256") != manifest_sha256():
        raise PrepareError(f"{out} was prepared for another manifest.json: prepare it again")
    if sha256(out / TASKS_FILE) != record.get("tasks_sha256"):
        raise PrepareError(f"{out / TASKS_FILE} does not match {PREPARED_FILE}")
    bad = [rel for rel, digest in record.get("files", {}).items()
           if not (out / rel).is_file() or sha256(out / rel) != digest]
    if bad:
        raise PrepareError(f"{len(bad)} prepared file(s) missing or changed since prepare, e.g. {bad[:3]}")
    return record


def _gold_fail_reason(g: Any) -> str:
    if g.modification["total"] == 0:
        return "the golden changes no answer cell, so V2's modification accuracy is 0.0"
    return (f"the golden fails against itself ({g.regression['correct']}/{g.regression['total']} "
            f"unchanged, {g.modification['correct']}/{g.modification['total']} changed): "
            f"{g.error_message[:200]}")


def sanity_grades(out: Path, *, grade=None) -> dict[str, Any]:
    """V2's grade of each task's golden against itself and of its input, one task at a
    time: the tasks whose golden fails (excluded by the Task, `excluded_ids`) and those
    whose input already passes (kept, listed). A grader failure (an unreadable file, memory
    running out) stops the check: it is the machine's, never a reason to drop a task."""
    from . import evaluation_port

    grade = grade or evaluation_port.grade
    out = Path(out)
    gold_fails, input_passes = [], []
    tasks = json.loads((out / TASKS_FILE).read_text(encoding="utf-8"))
    for n, task in enumerate(tasks, 1):
        _log(f"[{n}/{len(tasks)}] grading {task['id']}")
        inp, gold = str(out / task["input"]), str(out / task["golden"])
        try:
            g = grade(inp, gold, gold, task["answer_position"])
            if not g.passed:
                gold_fails.append({"id": task["id"], "reason": _gold_fail_reason(g)})
            if grade(inp, gold, inp, task["answer_position"]).passed:
                input_passes.append(task["id"])
        except evaluation_port.GraderFailed as exc:
            raise PrepareError(f"the grader failed on {task['id']}: {exc}") from exc
    return {"tasks_sha256": sha256(out / TASKS_FILE), "graded": len(tasks),
            "gold_fails": gold_fails, "input_passes": input_passes}


def check_grades(out: Path, *, grade=None) -> dict[str, Any]:
    """`check`, then `sanity_grades`, recorded in ``prepared.json`` (``sanity``): what the
    Task requires before it runs (`excluded_ids`)."""
    record = check(out)
    record["sanity"] = sanity_grades(out, grade=grade)
    (Path(out) / PREPARED_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return record


def excluded_ids(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    """The tasks the Task leaves out, by id, with why: the manifest's ``excluded`` (the
    lab's check, committed) and this directory's own ``check --grades`` (``sanity``), whose
    goldens fail against themselves. Refuses a directory with no current check."""
    sanity = record.get("sanity")
    if not isinstance(sanity, dict) or sanity.get("tasks_sha256") != record.get("tasks_sha256"):
        raise PrepareError("the prepared directory has no current `check --grades` (run "
                           "`python -m refactory_packages.spreadsheetbench_v2.prepare check "
                           "--grades OUT`): each golden must be graded against itself first")
    out = {row["id"]: row["reason"] for row in manifest.get("excluded", [])}
    out.update({row["id"]: row["reason"] for row in sanity.get("gold_fails", [])})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m refactory_packages.spreadsheetbench_v2.prepare")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="recalculate and freeze the release into OUT")
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    c = sub.add_parser("check", help="verify OUT's checksums")
    c.add_argument("out", type=Path)
    c.add_argument("--grades", action="store_true",
                   help="also grade each golden against itself and each input (slow)")
    m = sub.add_parser("manifest", help="(maintainers) rebuild manifest.json from the release")
    m.add_argument("source", type=Path)
    m.add_argument("--archive", type=Path, help="the release zip, for its checksum")
    m.add_argument("--grader", type=Path, help="V2's evaluation/evaluation.py, for its checksum")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            record = prepare(args.source, args.out)
            _log(f"prepared {len(record['files'])} files in {args.out}")
        elif args.command == "check":
            record = check_grades(args.out) if args.grades else check(args.out)
            if args.grades:
                for row in record["sanity"]["gold_fails"]:
                    _log(f"excluded {row['id']}: {row['reason']}")
                _log(json.dumps(record["sanity"], indent=1))
            _log(f"{args.out}: {len(record['files'])} files match")
        else:
            manifest = build_manifest(args.source, grader=args.grader,
                                      archive_sha256=sha256(args.archive) if args.archive else "")
            manifest_path().write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
            _log(f"wrote {manifest_path()}: {len(manifest['tasks'])} tasks, {len(manifest['dropped'])} dropped")
    except PrepareError as exc:
        _log(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
