"""SkillOpt's SpreadsheetBench target agent, ported line for line.

Everything here is copied from microsoft/SkillOpt (MIT License, Copyright (c) 2026
Microsoft Corporation) at commit ec9d021, ``skillopt/envs/spreadsheetbench/``, with the
source lines named on each piece. The graph's ops (`ops.py`), the Task (`tasks.py`) and
the parity checks (`parity.py`) all call these, so there is one copy of each behaviour.

What is not SkillOpt's: where generated code runs (`sandbox.py`: SkillOpt runs it
unsandboxed as the trainer's user, and its gold file is readable next to the input),
which interpreter runs it (`interpreter.py`), and the LibreOffice recalculation
(`recalc.py`, the lab's ``lo_grader.py``).
"""
from __future__ import annotations

import datetime
import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable, Sequence

import openpyxl

_HERE = Path(__file__).resolve().parent

#: ``prompts/codegen_system.md``, read verbatim: one line, no trailing newline
#: (``load_prompt`` does not strip it, skillopt/prompts/__init__.py:21-29).
CODEGEN_SYSTEM = (_HERE / "skills" / "codegen_system.md").read_text(encoding="utf-8")
#: ``skills/initial.md``, the starting skill (configs/spreadsheetbench/default.yaml:24).
INITIAL_SKILL = (_HERE / "skills" / "initial.md").read_text(encoding="utf-8")


# ── prompts: codegen_agent.py ──────────────────────────────────────────────

# codegen_agent.py:139-143 ``_build_system``
def skill_section(skill: Any) -> str:
    """What ``_build_system`` appends to the system file for a skill: ``\\n\\n## Skill\\n``
    and the stripped skill, or nothing for an empty one."""
    text = str(skill or "")
    return f"\n\n## Skill\n{text.strip()}" if text.strip() else ""


def build_system(skill: Any) -> str:
    """``_build_system``: the system file, then the skill section."""
    return CODEGEN_SYSTEM + skill_section(skill)


# codegen_agent.py:88-119 ``_preview_workbook``
def preview_workbook(path: str, max_rows: int = 5, max_cols: int = 20) -> str:
    """A text preview of the first few rows of each sheet."""
    wb = openpyxl.load_workbook(path, data_only=False)
    chunks: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        chunks.append(
            f"## Sheet: {sheet_name}  "
            f"(dim={ws.dimensions}, max_row={ws.max_row}, max_col={ws.max_column})"
        )
        for row in ws.iter_rows(
            min_row=1,
            max_row=min(ws.max_row, max_rows),
            max_col=min(ws.max_column, max_cols),
            values_only=False,
        ):
            cells = []
            for cell in row:
                v = cell.value
                if v is None:
                    cells.append(f"{cell.coordinate}=")
                else:
                    s = str(v)
                    if len(s) > 40:
                        s = s[:37] + "..."
                    cells.append(f"{cell.coordinate}={s}")
            chunks.append(" | ".join(cells))
        if ws.max_row > max_rows:
            chunks.append(f"... ({ws.max_row - max_rows} more rows)")
        chunks.append("")
    wb.close()
    return "\n".join(chunks)


# codegen_agent.py:146-191 ``_build_user``, with the diagnostic branches off (the trainer
# never passes ``diagnostic_mode``; port spec §1.3).
def build_user(instruction: str, input_xlsx: str, instruction_type: str = "",
               answer_position: str = "") -> str:
    """The first user message: instruction, type, answer position, preview, task."""
    try:
        preview = preview_workbook(input_xlsx)
    except Exception as e:  # noqa: BLE001
        preview = f"(failed to preview workbook: {e})"
    extra = ""
    if instruction_type:
        extra += f"\nInstruction type: {instruction_type}"
    if answer_position:
        extra += f"\nExpected answer position: {answer_position}"
    task_suffix = "Return only a ```python``` code block."
    return (
        f"# Instruction\n{instruction}\n{extra}\n\n"
        f"# Input spreadsheet preview\n{preview}\n\n"
        "# Task\n"
        "Write a Python script that reads the workbook from the variable `INPUT_PATH`, "
        "applies the instruction, and writes the modified workbook to `OUTPUT_PATH`. "
        "Preserve all other cells unchanged. "
        "The preview may be truncated — do not hardcode row counts or assume the data ends at the last previewed row; "
        "iterate over all actual rows in the workbook instead. "
        f"{task_suffix}"
    )


# codegen_agent.py:124-133 ``extract_code``
def extract_code(text: str) -> str:
    """The first fenced block of any language; with no usable fence, the whole reply."""
    if "```" not in text:
        return text.strip()
    start = text.find("```")
    nl = text.find("\n", start)
    end = text.find("```", nl + 1)
    if nl == -1 or end == -1:
        return text.strip()
    return text[nl + 1 : end].strip()


# codegen_agent.py:720-723
NO_CODE = (
    "No Python code block was found in your response. "
    "Please return a complete Python script inside a ```python``` block."
)


# codegen_agent.py:756-761
def exec_error_feedback(err: str) -> str:
    return (
        f"The code raised an error during execution:\n\n"
        f"```\n{err[:3000]}\n```\n\n"
        f"Please fix the code and return a complete corrected Python script "
        f"inside a ```python``` block."
    )


# codegen_agent.py:46-83 ``_build_eval_feedback``
def build_eval_feedback(verify_report: str) -> str:
    """Target feedback from a verify report, with the expected values removed."""
    wrong_lines = []
    n_correct = 0
    for raw_line in verify_report.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        m = re.match(
            r"(\S+!?\w+):\s*got=(.+?),\s*expected=.+?\s*(✓|✗)$",
            raw_line,
        )
        if m:
            cell, got_val, mark = m.groups()
            if mark == "✗":
                wrong_lines.append(f"  {cell}: your output = {got_val} (WRONG)")
            else:
                n_correct += 1
    lines = ["Your code executed successfully but produced incorrect results.",
             "The following cells have wrong values:"]
    lines.extend(wrong_lines)
    if n_correct:
        lines.append(f"  ({n_correct} other cells are correct.)")
    lines.append(
        "\nPlease analyze the spreadsheet data more carefully and fix the code. "
        "Return a complete corrected Python script inside a ```python``` block."
    )
    return "\n".join(lines)


# ── code execution: executor.py ────────────────────────────────────────────

# executor.py:12-23
RUNNER_TEMPLATE = textwrap.dedent(
    """
    import os, sys, traceback
    INPUT_PATH = {input_path!r}
    OUTPUT_PATH = {output_path!r}
    try:
    {user_code_indented}
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    """
)

# executor.py:27-29
_PATH_ASSIGN_RE = re.compile(
    r'^\s*(INPUT_PATH|OUTPUT_PATH)\s*=\s*.+$', re.MULTILINE
)

# executor.py:31-49, without PYTHONPATH and PYTHONHOME: SkillOpt's runs passed their
# own (the lab's pandas directory); here they would point the pinned interpreter at
# the host's packages, so the interpreter's own install stands alone.
_GENERATED_CODE_ENV_PASSTHROUGH = (
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SYSTEMDRIVE",
    "PATHEXT",
    "COMSPEC",
)


# executor.py:52-54
def strip_path_assignments(code: str) -> str:
    return _PATH_ASSIGN_RE.sub("", code)


# executor.py:57-89 (the POSIX part)
def generated_code_env(work_dir: str, temp_dir: str) -> dict[str, str]:
    """The minimal environment for generated code: PATH, HOME, TMPDIR, locale."""
    private_dir = os.path.abspath(work_dir or os.getcwd())
    private_temp = os.path.abspath(temp_dir)
    safe_env = {
        "PATH": os.environ.get("PATH") or os.defpath,
        "HOME": private_dir,
        "TMPDIR": private_temp,
    }
    for key in _GENERATED_CODE_ENV_PASSTHROUGH:
        value = os.environ.get(key)
        if value:
            safe_env[key] = value
    return {key: value for key, value in safe_env.items() if value}


def runner_script(code: str, input_path: str, output_path: str) -> str:
    """executor.py:94-101: the runner around the code."""
    cleaned = strip_path_assignments(code)
    indented = textwrap.indent(cleaned, "    ")
    return RUNNER_TEMPLATE.format(
        input_path=input_path,
        output_path=output_path,
        user_code_indented=indented,
    )


#: Turns ``[python, runner]`` into the command that runs it confined (`sandbox.wrap`).
Wrap = Callable[[list[str], str, str], list[str]]


# executor.py:92-126 ``run_generated_code``
def run_generated_code(code: str, input_path: str, output_path: str, timeout: int | None = 120,
                       *, python: str, wrap: Wrap | None = None,
                       cwd: str | None = None) -> tuple[bool, str]:
    """Run the code against ``input_path``; ``(ok, error)``. ``ok`` is exit status 0 with
    ``output_path`` present. ``python`` is the pinned interpreter (SkillOpt's is its own
    ``sys.executable``); ``wrap(command, writable_dir, temp_dir)`` confines it."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    script = runner_script(code, input_path, output_path)
    with tempfile.TemporaryDirectory(
        prefix="skillopt-generated-", ignore_cleanup_errors=True
    ) as temp_dir:
        runner = os.path.join(temp_dir, "runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(script)
        safe_env = generated_code_env(output_dir, temp_dir)
        command = [python, runner]
        ready_r = ready_w = None
        if wrap is not None:
            if getattr(wrap, "confirms_start", False):
                # Not SkillOpt's: the sandbox says on this pipe when the code starts, so
                # a setup failure is the harness's (`sandbox.SandboxFailed`), not the
                # model's code error.
                ready_r, ready_w = os.pipe()
                command = wrap(command, output_dir, temp_dir, ready_fd=ready_w)
            else:
                command = wrap(command, output_dir, temp_dir)
        try:
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout if timeout and timeout > 0 else None,
                    env=safe_env,
                    cwd=cwd,
                    pass_fds=(ready_w,) if ready_w is not None else (),
                )
            except subprocess.TimeoutExpired:
                _require_started(ready_r, ready_w, "")
                return False, f"timeout after {timeout}s"
            _require_started(ready_r, ready_w, proc.stderr)
            if proc.returncode != 0:
                return False, (proc.stdout + "\n" + proc.stderr).strip()
            if not os.path.exists(output_path):
                return False, "output file was not created"
            return True, ""
        finally:
            for fd in (ready_r, ready_w):
                if fd is not None:
                    os.close(fd)


def _require_started(ready_r: int | None, ready_w: int | None, stderr: str) -> None:
    """Raise `sandbox.SandboxFailed` when a sandbox that confirms its start never did."""
    if ready_r is None or ready_w is None:
        return
    os.set_blocking(ready_r, False)  # the write end stays open here: no EOF to wait for
    try:
        said = os.read(ready_r, 2)  # the sandbox's 2 bytes; nothing the code writes after
    except BlockingIOError:
        said = b""
    if said != b"ok":
        from refactory_dsh.model import HARNESS_INCIDENT_TEXT

        from .sandbox import SandboxFailed

        raise SandboxFailed(f"{HARNESS_INCIDENT_TEXT}the model-code sandbox failed to start: "
                            f"{stderr.strip()[-500:]}")


# ── grading: evaluator.py (a port of the official evaluator) ───────────────

# evaluator.py:28-31
def _datetime_to_float(dt: datetime.datetime) -> float:
    excel_start_date = datetime.datetime(1899, 12, 30)
    delta = dt - excel_start_date
    return delta.days + delta.seconds / 86400.0


# evaluator.py:35-52
def _transform_value(v):
    if isinstance(v, bool):
        return round(float(v), 2)
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, datetime.time):
        return str(v)[:-3]
    if isinstance(v, datetime.datetime):
        return round(_datetime_to_float(v), 0)
    if isinstance(v, str):
        try:
            return round(float(v), 2)
        except ValueError:
            return v
    return v


# evaluator.py:55-64
def compare_cell_value(v1, v2) -> bool:
    v1 = _transform_value(v1)
    v2 = _transform_value(v2)
    if (v1 == "" and v2 is None) or (v1 is None and v2 == ""):
        return True
    if (v1 == "" and v2 == "") or (v1 is None and v2 is None):
        return True
    if type(v1) is not type(v2):
        return False
    return v1 == v2


# evaluator.py:69-81
def _col_num2name(n: int) -> str:
    name = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        name = chr(65 + r) + name
    return name


def _col_name2num(name: str) -> int:
    num = 0
    for c in name:
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


# evaluator.py:84-98
def _parse_range(range_str: str):
    start_cell, end_cell = range_str.split(":")
    sc = "".join(ch for ch in start_cell if ch.isalpha())
    sr = "".join(ch for ch in start_cell if ch.isdigit())
    ec = "".join(ch for ch in end_cell if ch.isalpha())
    er = "".join(ch for ch in end_cell if ch.isdigit())
    return (_col_name2num(sc), int(sr)), (_col_name2num(ec), int(er))


def generate_cell_names(range_str: str):
    if ":" not in range_str:
        return [range_str]
    (sc, sr), (ec, er) = _parse_range(range_str)
    cols = [_col_num2name(i) for i in range(sc, ec + 1)]
    return [f"{c}{r}" for c in cols for r in range(sr, er + 1)]


# evaluator.py:101-111
def _cell_level_compare(wb_gt, wb_proc, sheet_name: str, cell_range: str):
    if sheet_name not in wb_proc.sheetnames:
        return False, f"worksheet not found: {sheet_name}"
    ws_gt = wb_gt[sheet_name]
    ws_proc = wb_proc[sheet_name]
    for cn in generate_cell_names(cell_range):
        cg = ws_gt[cn]
        cp = ws_proc[cn]
        if not compare_cell_value(cg.value, cp.value):
            return False, f"value@{sheet_name}!{cn}: gt={cg.value!r} pred={cp.value!r}"
    return True, ""


# evaluator.py:116-147
def compare_workbooks(gt_file: str, proc_file: str, answer_position: str) -> tuple[bool, str]:
    """Return (ok, msg). Single test-case comparison, official semantics."""
    if not os.path.exists(proc_file):
        return False, "file not exist"
    try:
        wb_gt = openpyxl.load_workbook(filename=gt_file, data_only=True)
        wb_proc = openpyxl.load_workbook(filename=proc_file, data_only=True)
    except Exception as e:  # noqa: BLE001
        return False, f"load error: {e}"
    try:
        ok_all = True
        msg_first = ""
        for scr in (answer_position or "").split(","):
            scr = scr.strip()
            if not scr:
                continue
            if "!" in scr:
                sheet_name, cell_range = scr.split("!", 1)
                sheet_name = sheet_name.strip().strip("'\"")
            else:
                sheet_name = wb_gt.sheetnames[0]
                cell_range = scr
            cell_range = cell_range.strip().strip("'\"")
            ok, msg = _cell_level_compare(wb_gt, wb_proc, sheet_name, cell_range)
            if not ok:
                ok_all = False
                if not msg_first:
                    msg_first = msg
        return ok_all, msg_first
    finally:
        wb_gt.close()
        wb_proc.close()


# evaluator.py:150-158
def evaluate_cached(pred_path: str, gold_path: str, instruction_type: str,
                    answer_position: str) -> dict:
    """SkillOpt's grader as shipped: cached cell values only."""
    ok, msg = compare_workbooks(gold_path, pred_path, answer_position)
    return {"ok": ok, "reason": msg, "instruction_type": instruction_type}


# rollout.py:112-211 ``_auto_verify_output``
_MAX_REPORT_CHARS = 12000


def auto_verify_output(pred_path: str, gold_path: str, answer_position: str) -> str:
    """The per-cell verification report (``got=..., expected=... ✓|✗``), head+tail
    truncated at 12,000 characters."""
    if not os.path.exists(pred_path):
        return "Verification: output file does not exist."
    try:
        wb_pred = openpyxl.load_workbook(pred_path, data_only=True)
        wb_gold = openpyxl.load_workbook(gold_path, data_only=True)
    except Exception as e:
        return f"Verification: could not open workbooks: {e}"

    lines = ["## Output Verification"]
    try:
        for scr in (answer_position or "").split(","):
            scr = scr.strip()
            if not scr:
                continue
            if "!" in scr:
                sheet_name, cell_range = scr.split("!", 1)
                sheet_name = sheet_name.strip().strip("'\"")
            else:
                sheet_name = wb_gold.sheetnames[0]
                cell_range = scr
            cell_range = cell_range.strip().strip("'\"")

            cell_names = generate_cell_names(cell_range)
            ws_pred = wb_pred[sheet_name] if sheet_name in wb_pred.sheetnames else None
            ws_gold = wb_gold[sheet_name] if sheet_name in wb_gold.sheetnames else None

            if ws_pred is None:
                lines.append(f"  Sheet '{sheet_name}' NOT FOUND in output.")
                continue

            n_empty_correct = 0
            for cn in cell_names:
                gv = ws_gold[cn].value if ws_gold else "N/A"
                pv = ws_pred[cn].value
                ok_cell = ws_gold is not None and compare_cell_value(gv, pv)
                if ok_cell and gv in (None, "") and pv in (None, ""):
                    n_empty_correct += 1
                    continue
                match = "✓" if ok_cell else "✗"
                lines.append(f"  {sheet_name}!{cn}: got={pv!r}, expected={gv!r} {match}")
            if n_empty_correct:
                lines.append(
                    f"  (+{n_empty_correct} empty cells correct, omitted)"
                )

        formula_cells = []
        for sn in wb_pred.sheetnames:
            ws = wb_pred[sn]
            for row in ws.iter_rows(max_row=min(ws.max_row, 200), values_only=False):
                for cell in row:
                    if isinstance(cell.value, str) and cell.value.startswith("="):
                        formula_cells.append(f"{sn}!{cell.coordinate}={cell.value}")
                        if len(formula_cells) >= 10:
                            break
                if len(formula_cells) >= 10:
                    break
            if len(formula_cells) >= 10:
                break
        if formula_cells:
            lines.append(f"\n  WARNING: {len(formula_cells)} cells contain Excel formulas (openpyxl cannot evaluate them):")
            for fc in formula_cells[:5]:
                lines.append(f"    {fc}")
            if len(formula_cells) > 5:
                lines.append(f"    ... and {len(formula_cells) - 5} more")
    finally:
        wb_pred.close()
        wb_gold.close()

    report = "\n".join(lines)
    if len(report) > _MAX_REPORT_CHARS:
        half = _MAX_REPORT_CHARS // 2
        report = (
            report[:half]
            + f"\n  ...[verification report truncated, {len(report)} chars total]...\n"
            + report[-half:]
        )
    return report


# ── item data: rollout.py ───────────────────────────────────────────────────

# rollout.py:62-91 ``_find_test_cases``
def find_test_cases(task_dir: str) -> list[tuple[str, str, str]]:
    """``[(case_no, input_path, answer_path), ...]``: ``*_input``/``*_answer``, then
    ``*_init``/``*_golden``, then bare ``initial``/``golden``."""
    import glob as _glob

    cases: list[tuple[str, str, str]] = []
    inputs = sorted(_glob.glob(os.path.join(task_dir, "*_input.xlsx")))
    for ip in inputs:
        no = os.path.basename(ip).split("_", 1)[0]
        ap = ip.replace("_input.xlsx", "_answer.xlsx")
        if os.path.exists(ap):
            cases.append((no, ip, ap))
    inits = sorted(_glob.glob(os.path.join(task_dir, "*_init.xlsx")))
    for ip in inits:
        no = os.path.basename(ip).split("_", 1)[0]
        ap = ip.replace("_init.xlsx", "_golden.xlsx")
        if os.path.exists(ap):
            cases.append((no, ip, ap))
    if not cases:
        bare_init = os.path.join(task_dir, "initial.xlsx")
        bare_gold = os.path.join(task_dir, "golden.xlsx")
        if os.path.exists(bare_init) and os.path.exists(bare_gold):
            cases.append(("1", bare_init, bare_gold))
    return cases


# rollout.py:638-650
def answer_position_eval(item: dict) -> str:
    """The answer position the model sees and the grader reads: prefixed with the
    item's answer sheet when it names none."""
    answer_position = item.get("answer_position", "")
    answer_sheet = item.get("answer_sheet", "")
    if answer_position and answer_sheet and "!" not in answer_position:
        return f"{answer_sheet}!{answer_position}"
    return answer_position


def task_type(instruction_type: str) -> str:
    itype_lower = (instruction_type or "").lower()
    if "cell" in itype_lower:
        return "cell_level"
    if "sheet" in itype_lower:
        return "sheet_level"
    return "other"


def tail_line(err: str) -> str:
    """rollout.py:806: the last line of an error, for ``exec-error: <line>``."""
    return err.strip().splitlines()[-1][:200] if err.strip() else "unknown"


__all__: Sequence[str] = (
    "CODEGEN_SYSTEM", "INITIAL_SKILL", "NO_CODE", "answer_position_eval",
    "auto_verify_output", "build_eval_feedback", "build_system", "build_user",
    "compare_cell_value", "compare_workbooks", "evaluate_cached", "exec_error_feedback",
    "extract_code", "find_test_cases", "generate_cell_names", "preview_workbook",
    "run_generated_code", "runner_script", "skill_section", "task_type",
)
