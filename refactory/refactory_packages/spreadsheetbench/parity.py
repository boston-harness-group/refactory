"""The free parity checks (port spec §6.1-6.5) and the start-skill comparison (§6.7).

Each check reads a real SkillOpt run directory (``runs/so-s42`` on the lab: its
``calls.jsonl`` and ``out/``) and the release data (``SB_DATA``), and makes no model
call::

    python -m refactory_packages.spreadsheetbench.parity prompts  --run <so-s42>
    python -m refactory_packages.spreadsheetbench.parity executor --run <so-s42> [--skillopt <checkout>] [--limit N]
    python -m refactory_packages.spreadsheetbench.parity grader   --run <so-s42> [--grader cached|recalc] [--rescore <lo-test-rescore.json>]
    python -m refactory_packages.spreadsheetbench.parity feedback --run <so-s42>
    python -m refactory_packages.spreadsheetbench.parity startskill --reference <grader.json> ... --port <port.json> ...

``startskill`` takes SkillOpt's side from ``grader --grader recalc`` outputs of its runs
(the val start-skill pass, ``selection_eval_baseline``), one per seed, and the port's
side as ``{"scores": [...]}``.

Every check prints one JSON summary and exits 1 on any mismatch. The reference numbers
of ``startskill`` are an input (SkillOpt's passes at whatever item budget they ran),
never a constant here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from refactory_packages.skillopt.claude_backend import claude_backend_messages

from . import recalc
from . import skillopt_port as port

TARGET_SYSTEM_HEAD = "You are an expert Python programmer"


def _pass_dirs(run: Path) -> list[Path]:
    """Every pass directory of a SkillOpt run that holds ``predictions/``."""
    return sorted(Path(p).parent for p in glob.glob(str(run / "out" / "**" / "predictions"),
                                                    recursive=True))


def _rows(data: Path) -> dict[str, dict[str, Any]]:
    return {str(r["id"]): r for r in json.loads((data / "dataset.json").read_text())}


def _conversation(item_dir: Path) -> list[dict[str, Any]] | None:
    path = item_dir / "conversation.json"
    if not path.exists():
        return None
    return [m for m in json.loads(path.read_text()) if m.get("role") in ("user", "assistant")]


def check_prompts(run: Path, data: Path) -> dict[str, Any]:
    """§6.1: every recorded target call's system and prompt against the port's, built
    from the messages SkillOpt saved (its first user message and its conversation); and
    the port's first user message, preview and initial-skill system against the files
    each item saved, rebuilt from the release data."""
    calls = [json.loads(line) for line in (run / "calls.jsonl").read_text().splitlines() if line]
    target = [c for c in calls if str(c.get("system", "")).startswith(TARGET_SYSTEM_HEAD)]
    built: dict[str, set[str]] = {}
    rows = _rows(data)
    rebuilt = {"items": 0, "user_equal": 0, "preview_equal": 0}
    initial = {"items": 0, "system_equal": 0}
    rebuilt_bad: list[str] = []
    seen_items: set[tuple[str, str]] = set()
    for pass_dir in _pass_dirs(run):
        for item_dir in sorted((pass_dir / "predictions").iterdir()):
            user_file = item_dir / "target_user_prompt.txt"
            if not user_file.exists():
                continue
            user = user_file.read_text()
            system = (item_dir / "target_system_prompt.txt").read_text()
            conv = _conversation(item_dir) or []
            for k in range(0, len(conv) + 1, 2):
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": user}, *conv[:k]]
                sys_text, prompt = claude_backend_messages(messages)
                built.setdefault(prompt, set()).add(sys_text)
            if "baseline" in pass_dir.name or "initial" in pass_dir.name:
                # A pass of SkillOpt's starting skill: its system file is the port's.
                initial["items"] += 1
                initial["system_equal"] += system == port.build_system(port.INITIAL_SKILL)
            item = rows.get(item_dir.name)
            key = (item_dir.name, user)
            if item is None or key in seen_items:
                continue
            seen_items.add(key)
            cases = port.find_test_cases(str(data / item["spreadsheet_path"]))
            if not cases:
                continue
            rebuilt["items"] += 1
            mine = port.build_user(item["instruction"], cases[0][1], item.get("instruction_type", ""),
                                   port.answer_position_eval(item))
            rebuilt["user_equal"] += mine == user
            preview = item_dir / "spreadsheet_preview.txt"
            rebuilt["preview_equal"] += (not preview.exists()
                                         or port.preview_workbook(cases[0][1]) == preview.read_text())
            if mine != user:
                rebuilt_bad.append(item_dir.name)
    matched = sum(1 for c in target if c["system"] in built.get(c["prompt"], ()))
    unmatched = [c for c in target if c["prompt"] not in built]
    return {
        "target_calls": len(target),
        "system_and_prompt_equal": matched,
        "prompt_equal_system_differs": sum(1 for c in target if c["prompt"] in built
                                           and c["system"] not in built[c["prompt"]]),
        "no_saved_messages": len(unmatched),
        "no_saved_messages_turn1": sum(1 for c in unmatched if "\n- Assistant:" not in c["prompt"]),
        "rebuilt_from_data": rebuilt,
        "initial_skill_passes": initial,
        "rebuilt_differs": rebuilt_bad[:20],
        "ok": (matched + len(unmatched) == len(target)
               and not any("\n- Assistant:" not in c["prompt"] for c in unmatched)
               and rebuilt["user_equal"] == rebuilt["items"]
               and rebuilt["preview_equal"] == rebuilt["items"]
               and initial["system_equal"] == initial["items"]),
    }


_SKILLOPT_RUN = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("so_executor", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ok, err = mod.run_generated_code(open(sys.argv[2]).read(), sys.argv[3], sys.argv[4])
print(json.dumps([ok, err]))
"""


def _cell_text(value: Any) -> str:
    """A cell value as comparable text: an array or data-table formula by its text and
    range (its ``repr`` names the object's address)."""
    if hasattr(value, "text") and hasattr(value, "ref"):
        return f"{type(value).__name__}({value.ref!r}, {value.text!r})"
    return repr(value)


def _cells(path: str) -> dict[str, Any] | None:
    import openpyxl

    if not os.path.exists(path):
        return None
    wb = openpyxl.load_workbook(path, data_only=False)
    try:
        return {f"{ws.title}!{c.coordinate}": _cell_text(c.value) for ws in wb.worksheets
                for row in ws.iter_rows() for c in row if c.value is not None}
    finally:
        wb.close()


_TEMP = re.compile(r"/[^\s\"']*(?:skillopt-generated-|sb-item-|sbx-)[^\s\"']*")


def check_executor(run: Path, data: Path, skillopt: Path | None, limit: int) -> dict[str, Any]:
    """§6.3: so-s42's generated code run by the port (pinned interpreter, sandbox) and,
    with ``--skillopt``, by SkillOpt's own ``run_generated_code`` under the same
    interpreter, on the same input: success, error text (temporary paths masked) and
    every output cell must agree."""
    from . import interpreter, sandbox

    python = interpreter.interpreter()
    sandbox.require(python)
    rows = _rows(data)
    hidden = sandbox.hidden_paths([str(data), str(run)])
    so_executor = (skillopt / "skillopt/envs/spreadsheetbench/executor.py") if skillopt else None
    checked = agree = 0
    differ: list[dict[str, Any]] = []
    samples = sorted(glob.glob(str(run / "out" / "test_eval_baseline" / "predictions" / "*" / "code.py")))
    for code_file in samples[:limit]:
        item = rows.get(Path(code_file).parent.name)
        cases = port.find_test_cases(str(data / item["spreadsheet_path"])) if item else []
        code = Path(code_file).read_text()
        if not cases or not code.strip():
            continue
        checked += 1
        with tempfile.TemporaryDirectory(prefix="sbx-") as work:
            root = Path(work)
            for side in ("port", "so"):
                (root / side / "in").mkdir(parents=True)
                (root / side / "out").mkdir()
                (root / side / "in" / Path(cases[0][1]).name).write_bytes(Path(cases[0][1]).read_bytes())
            mine = port.run_generated_code(
                code, str(root / "port/in" / Path(cases[0][1]).name), str(root / "port/out/1_pred.xlsx"),
                python=python, wrap=sandbox.wrapper(hidden, visible=[root / "port"]),
                cwd=str(root / "port"))
            result: dict[str, Any] = {"id": item["id"], "port": [mine[0], _TEMP.sub("<tmp>", mine[1])[-300:]]}
            same = True
            if so_executor is not None:
                out = subprocess.run(
                    [python, "-c", _SKILLOPT_RUN, str(so_executor), code_file,
                     str(root / "so/in" / Path(cases[0][1]).name), str(root / "so/out/1_pred.xlsx")],
                    capture_output=True, text=True, timeout=300, cwd=str(root / "so"))
                theirs = json.loads(out.stdout.strip().splitlines()[-1])
                result["skillopt"] = [theirs[0], _TEMP.sub("<tmp>", theirs[1])[-300:]]
                same = (mine[0] == theirs[0]
                        and _TEMP.sub("<tmp>", mine[1]).replace("/port/", "/x/")
                        == _TEMP.sub("<tmp>", theirs[1]).replace("/so/", "/x/")
                        and _cells(str(root / "port/out/1_pred.xlsx"))
                        == _cells(str(root / "so/out/1_pred.xlsx")))
            agree += same
            if not same:
                differ.append(result)
    return {"checked": checked, "agree": agree, "differ": differ[:20],
            "compared_with_skillopt": so_executor is not None, "ok": agree == checked}


def check_grader(run: Path, data: Path, choice: str, rescore: Path | None) -> dict[str, Any]:
    """§6.4: every saved prediction regraded by the port. Under ``cached`` each must
    equal its pass's ``results.jsonl`` ``hard``; under ``recalc`` the passing count of
    each pass named in ``--rescore`` (the lab's ``lo-test-rescore-*.json``) must equal
    its ``fair``. Items SkillOpt scored as a timeout or never graded are counted apart."""
    rows = _rows(data)
    expected_fair: dict[str, int] = {}
    if rescore is not None:
        seed = re.search(r"so-s(\d+)", str(run))
        for key, value in json.loads(rescore.read_text()).items():
            label, _, name = key.partition("/")
            if seed and label == f"s{seed.group(1)}":
                expected_fair[name] = int(value["fair"])
    passes: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for pass_dir in _pass_dirs(run):
        results_file = pass_dir / "results.jsonl"
        if not results_file.exists():
            continue
        name = str(pass_dir.relative_to(run / "out"))
        n = equal = passed = skipped = 0
        for line in results_file.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            item = rows.get(str(r["id"]))
            cases = port.find_test_cases(str(data / item["spreadsheet_path"])) if item else []
            pred = pass_dir / "predictions" / str(r["id"]) / f"{cases[0][0] if cases else 1}_pred.xlsx"
            if not cases or r.get("phase") in ("timeout", "error", "llm", "setup", "extract") \
                    or not pred.exists():
                skipped += 1
                passed += 0
                continue
            try:
                ev = recalc.evaluate(str(pred), cases[0][2], item.get("instruction_type", ""),
                                     port.answer_position_eval(item), choice=choice)
            except Exception as e:  # noqa: BLE001 - rollout.py:820-823
                ev = {"ok": False, "reason": f"eval-exception: {type(e).__name__}: {e}"}
            n += 1
            passed += bool(ev["ok"])
            if choice == "cached":
                equal += int(bool(ev["ok"])) == int(r.get("hard", 0))
                if int(bool(ev["ok"])) != int(r.get("hard", 0)):
                    mismatches.append({"pass": name, "id": r["id"], "port": ev["ok"],
                                       "skillopt": r.get("hard"), "reason": ev.get("reason", "")[:200]})
        passes[name] = {"graded": n, "port_pass": passed, "not_graded": skipped,
                        **({"equal": equal} if choice == "cached" else {}),
                        **({"expected_fair": expected_fair[name]} if name in expected_fair else {})}
    ok = not mismatches if choice == "cached" else all(
        v["port_pass"] == v["expected_fair"] for v in passes.values() if "expected_fair" in v)
    return {"grader": choice, "passes": passes, "mismatches": mismatches[:30], "ok": ok}


_EVAL_FEEDBACK_HEAD = "Your code executed successfully but produced incorrect results."


def check_feedback(run: Path, data: Path, choice: str) -> dict[str, Any]:
    """§6.5: the train feedback of every rollout whose conversation ends in eval feedback
    (its saved prediction is the one that feedback was built from), rebuilt by the port,
    against the recorded message."""
    rows = _rows(data)
    checked = equal = 0
    differ: list[dict[str, Any]] = []
    for pass_dir in _pass_dirs(run):
        for item_dir in sorted((pass_dir / "predictions").iterdir()):
            conv = _conversation(item_dir)
            if not conv or conv[-1]["role"] != "user" \
                    or not conv[-1]["content"].startswith(_EVAL_FEEDBACK_HEAD):
                continue
            item = rows.get(item_dir.name)
            cases = port.find_test_cases(str(data / item["spreadsheet_path"])) if item else []
            if not cases:
                continue
            pred = item_dir / f"{cases[0][0]}_pred.xlsx"
            mine = port.build_eval_feedback(recalc.auto_verify_output(
                str(pred), cases[0][2], port.answer_position_eval(item), choice=choice))
            checked += 1
            equal += mine == conv[-1]["content"]
            if mine != conv[-1]["content"]:
                differ.append({"dir": str(item_dir.relative_to(run)), "port": mine[:400],
                               "skillopt": conv[-1]["content"][:400]})
    return {"checked": checked, "equal": equal, "differ": differ[:10], "ok": equal == checked}


def _scores(path: Path, pass_name: str) -> list[float]:
    """Pass scores from a file: ``{"passes": [score, ...]}`` or ``{"scores": [...]}``
    (fractions), or a ``grader`` check's output, whose pass ``pass_name`` gives
    ``port_pass`` over all its items."""
    data = json.loads(path.read_text())
    if isinstance(data.get("passes"), dict):
        found = data["passes"].get(pass_name)
        if found is None:
            raise SystemExit(f"{path} has no pass {pass_name!r}")
        return [found["port_pass"] / (found["graded"] + found["not_graded"])]
    return [float(x) for x in data.get("passes") or data.get("scores") or []]


def check_startskill(references: list[Path], port_summaries: list[Path], pass_name: str,
                     sd_margin: float) -> dict[str, Any]:
    """§6.7 on val: the port's start-skill passes against SkillOpt's. Both sides are
    inputs (`_scores`), so the reference is whatever SkillOpt run it names -- at the item
    budget that run used -- never a constant here. Within noise means the gap between the
    means is at most ``sd_margin`` Welch standard errors."""
    ref = [x for path in references for x in _scores(path, pass_name)]
    mine = [x for path in port_summaries for x in _scores(path, pass_name)]
    ref_mean, mine_mean = statistics.mean(ref), statistics.mean(mine)
    # Welch's standard error, each side with its own spread; a side of one pass takes
    # the other side's.
    ref_sd = statistics.stdev(ref) if len(ref) > 1 else None
    mine_sd = statistics.stdev(mine) if len(mine) > 1 else None
    ref_sd = ref_sd if ref_sd is not None else (mine_sd or 0.0)
    mine_sd = mine_sd if mine_sd is not None else ref_sd
    se = (ref_sd ** 2 / len(ref) + mine_sd ** 2 / len(mine)) ** 0.5
    gap = mine_mean - ref_mean
    return {"reference": ref, "port": mine, "reference_mean": ref_mean, "port_mean": mine_mean,
            "gap": gap, "standard_error": se,
            "ok": abs(gap) <= sd_margin * se if se else gap == 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m refactory_packages.spreadsheetbench.parity")
    sub = parser.add_subparsers(dest="check", required=True)
    for name in ("prompts", "executor", "grader", "feedback"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True, help="a SkillOpt run directory (so-s42)")
        p.add_argument("--data", type=Path, default=Path(os.environ.get("SB_DATA", "")))
        if name == "executor":
            p.add_argument("--skillopt", type=Path, help="a SkillOpt checkout to compare with")
            p.add_argument("--limit", type=int, default=40)
        if name in ("grader", "feedback"):
            p.add_argument("--grader", choices=recalc.GRADERS, default="cached")
        if name == "grader":
            p.add_argument("--rescore", type=Path)
    p = sub.add_parser("startskill")
    p.add_argument("--reference", type=Path, action="append", required=True,
                   help="SkillOpt's passes; repeat for each seed")
    p.add_argument("--port", type=Path, action="append", required=True,
                   help="the port's passes; repeat for each")
    p.add_argument("--pass", dest="pass_name", default="selection_eval_baseline",
                   help="which pass of a grader-check output (SkillOpt's val start skill)")
    p.add_argument("--sd-margin", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.check == "prompts":
        out = check_prompts(args.run, args.data)
    elif args.check == "executor":
        out = check_executor(args.run, args.data, args.skillopt, args.limit)
    elif args.check == "grader":
        out = check_grader(args.run, args.data, args.grader, args.rescore)
    elif args.check == "feedback":
        out = check_feedback(args.run, args.data, args.grader)
    else:
        out = check_startskill(args.reference, args.port, args.pass_name, args.sd_margin)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
