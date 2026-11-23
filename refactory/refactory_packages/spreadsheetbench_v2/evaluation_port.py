"""SpreadsheetBench 2's grader (``evaluation/evaluation.py``), ported line for line.

Copied from the SpreadsheetBench 2 repository's ``evaluation/evaluation.py`` (the file the
release ships, sha256 in ``manifest.json``: ``grader_sha256``); each piece names the lines
it copies. The comparison is V2's exactly: numbers within 1% (relative; absolute against
0), ``None`` equal to 0 and to ``""``, dates as whole Excel day numbers, formatting
ignored, a cell holding an Excel error compared by its formula, and an item
correct when every changed cell matches and at least 99.8% of the unchanged ones do
(``process_single_item``, :452-458).

``compare_cell_value`` treats ``#DIV/0!``/``#N/A`` as equal to finance "not meaningful"
text (``N/A``, ``NM``, ``-``, ..., :44-61), but on the workbook path that rule does not
reach a real error cell: when either cell's cached value is an Excel error, V2 compares
the two cells' formulas instead (:298, :324), so a golden ``#DIV/0!`` (from ``=A2/0``)
never equals an output's ``"N/A"`` text (parity case ``not_meaningful``). The rule only
applies between texts such as ``"N/A"`` and ``"-"``, or inside that formula comparison.

Two things differ in how it runs, never in what it decides (the parity fixture
``tests/fixtures/spreadsheetbench_v2_grader_parity.json`` was recorded by calling V2's own
functions):

- **One workbook in memory at a time.** V2 loads the input, golden and output twice each
  (cached values and formulas), all six at once (:361-371); a Debugging-10 workbook then
  needs more than 3 GB, and V2's ``except Exception`` turns the ``MemoryError`` into a 0
  with an empty message (:372-373). Here each workbook is loaded, the answer cells' values
  are taken, and it is closed before the next; the formula view is loaded only when some
  answer cell holds an Excel error, the only time V2 reads it (:298, :324). V2's functions
  read nothing of a cell but its value (``compare_cell_value``, ``compare_cell_formula``),
  so the values decide alike.
- **A failure is never a silent 0.** A workbook that cannot be opened or read raises
  `GraderFailed` (a harness incident, with its message) when it is the prepared input or
  golden, or when memory ran out; an output the model wrote that cannot be opened is the
  model's failure, scored 0 with the reason (V2 scores it 0 too, :372-373).

Not ported: the ``with_font_color`` and ``with_formula`` modes (:276-279, :431-436), which
V2 turns on for the Debugging "Color" and "Embedded" inputs alone. Those 20 tasks are
dropped (``manifest.json``), so `grade` refuses a path that would need them.
"""
from __future__ import annotations

import datetime
import gc
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from refactory_dsh.model import HARNESS_INCIDENT_TEXT, HarnessIncident


class GraderFailed(HarnessIncident):
    """The grader could not compare: the prepared input or golden would not open or read,
    or memory ran out. The harness's failure, never the model's, so the item is a harness
    incident (run again, then left out of the score) with the message, never a 0."""


# evaluation.py:17-20
def datetime_to_float(dt):
    excel_start_date = datetime.datetime(1899, 12, 30)
    delta = dt - excel_start_date
    return delta.days + delta.seconds / 86400.0


# evaluation.py:23-35
def transform_value(v):
    if isinstance(v, (int, float)):
        v = round(float(v), 2)
    elif isinstance(v, datetime.time):
        v = str(v)[:-3]
    elif isinstance(v, datetime.datetime):
        v = round(datetime_to_float(v), 0)
    elif isinstance(v, str):
        try:
            v = round(float(v), 2)
        except ValueError:
            pass
    return v


# evaluation.py:44-51
_DISPLAY_EQUIVALENT_ERRORS = {'#DIV/0!', '#N/A'}
_NOT_MEANINGFUL = _DISPLAY_EQUIVALENT_ERRORS | {
    'N/A', 'NA', 'N.A.', 'N/M', 'NM', 'N.M.',
    'NOT MEANINGFUL', 'NOT APPLICABLE', 'NOT AVAILABLE',
    '—', '–', '-', '--', '---',
}


# evaluation.py:52-58
def _is_not_meaningful(v):
    if v is None:
        return False
    if isinstance(v, str):
        s = v.strip().upper()
        return s in _NOT_MEANINGFUL or s in _DISPLAY_EQUIVALENT_ERRORS
    return False


# evaluation.py:38-103
def compare_cell_value(v1, v2, tolerance=0.01):

    # Handle ArrayFormula objects — compare by formula text
    if hasattr(v1, 'text') and hasattr(v2, 'text'):
        return v1.text == v2.text

    if _is_not_meaningful(v1) and _is_not_meaningful(v2):
        return True

    # For two numeric values, compare on raw values with tolerance only
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if v1 == v2:
            return True
        if v1 == 0 or v2 == 0:
            return abs(v1 - v2) <= tolerance
        return abs(v1 - v2) / max(abs(v1), abs(v2)) <= tolerance

    v1 = transform_value(v1)
    v2 = transform_value(v2)
    if (v1 == "" and v2 is None) or (v1 is None and v2 == ""):
        return True
    if (v1 == "" and v2 == "") or (v1 is None and v2 is None):
        return True
    # Treat None as equivalent to 0
    if (v1 is None and isinstance(v2, (int, float)) and v2 == 0) or \
       (v2 is None and isinstance(v1, (int, float)) and v1 == 0):
        return True
    if type(v1) != type(v2):
        return False
    if v1 == v2:
        return True
    # Formula strings: compare case-insensitively
    if isinstance(v1, str) and isinstance(v2, str) and v1.startswith('=') and v2.startswith('='):
        return v1.replace('$', '').upper() == v2.replace('$', '').upper()
    # Numeric type: check 1% tolerance
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if v1 == 0 and v2 == 0:
            return True
        if v1 == 0 or v2 == 0:
            return abs(v1 - v2) <= tolerance
        relative_error = abs(v1 - v2) / max(abs(v1), abs(v2))
        return relative_error <= tolerance
    return False


# evaluation.py:161-185 ``compare_cell_formula``, on the two cells' values (all it reads
# of them, cell1.value and cell2.value at :163).
def compare_cell_formula(f1, f2) -> bool:
    if hasattr(f1, 'text') and hasattr(f2, 'text'):
        return f1.text == f2.text

    if isinstance(f1, str) and isinstance(f2, str) and f1.startswith('=') and f2.startswith('='):
        def _normalize_formula(f):
            f = f.replace('$', '').upper()
            if f.startswith('=+'):
                f = '=' + f[2:]
            return f
        return _normalize_formula(f1) == _normalize_formula(f2)

    empty = (None, "")
    if f1 in empty and f2 in empty:
        return True

    return compare_cell_value(f1, f2)


# evaluation.py:188-202
def parse_answer_position(answer_position):
    """Parse answer_position string, handling sheet names with commas."""
    result, current, in_quote = [], "", False
    for char in answer_position:
        if char == "'":
            in_quote = not in_quote
        if char == ',' and not in_quote:
            if current.strip():
                result.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        result.append(current.strip())
    return result


# evaluation.py:205-211
def col_num2name(n):
    name = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name


# evaluation.py:214-219
def col_name2num(name):
    num = 0
    for c in name:
        num = num * 26 + (ord(c) - ord('A') + 1)
    return num


# evaluation.py:222-239
def parse_cell_range(range_str):
    start_cell, end_cell = range_str.split(':')
    start_col, start_row = '', ''
    for char in start_cell:
        if char.isdigit():
            start_row += char
        else:
            start_col += char

    end_col, end_row = '', ''
    for char in end_cell:
        if char.isdigit():
            end_row += char
        else:
            end_col += char

    return (col_name2num(start_col), int(start_row)), (col_name2num(end_col), int(end_row))


# evaluation.py:242-249
def generate_cell_names(range_str):
    if ':' not in range_str:
        return [range_str]
    (start_col, start_row), (end_col, end_row) = parse_cell_range(range_str)
    columns = [col_num2name(i) for i in range(start_col, end_col + 1)]
    return [f"{col}{row}" for col in columns for row in range(start_row, end_row + 1)]


# evaluation.py:252
_EXCEL_ERRORS = {'#REF!', '#VALUE!', '#DIV/0!', '#NAME?', '#NULL!', '#N/A', '#NUM!'}


# evaluation.py:255-263
def _find_sheet(wb, name):
    if name in wb.sheetnames:
        return wb[name]
    name_stripped = name.strip().lower()
    for sn in wb.sheetnames:
        if sn.strip().lower() == name_stripped:
            return wb[sn]
    return None


# evaluation.py:266-269, on the cell's value
def _has_excel_error(v) -> bool:
    return isinstance(v, str) and v in _EXCEL_ERRORS


# ── the answer cells' values, one workbook at a time ──────────────────────────

#: One answer range: the sheet name as the position gives it, and its cells.
Range = tuple[str, list[str]]


def answer_ranges(answer_position: str, first_sheet: str) -> list[Range]:
    """evaluation.py:378-384: each range of the position, its sheet the golden's first
    when it names none."""
    ranges: list[Range] = []
    for sheet_cell_range in parse_answer_position(answer_position):
        if '!' in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.split('!')
        else:
            sheet_name, cell_range = first_sheet, sheet_cell_range
        sheet_name = sheet_name.strip("'").strip()
        cell_range = cell_range.strip("'").strip()
        ranges.append((sheet_name, generate_cell_names(cell_range)))
    return ranges


#: A workbook's view of the answer ranges: per range, ``None`` when it lacks the sheet
#: (`_find_sheet`), else each cell's value.
View = list["dict[str, Any] | None"]


def _load(path: str, data_only: bool):
    import openpyxl

    return openpyxl.load_workbook(filename=path, data_only=data_only)


def _first_sheet(path: str) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(filename=path, read_only=True)
    try:
        return wb.sheetnames[0]
    finally:
        wb.close()


def read_view(path: str, ranges: list[Range], *, data_only: bool,
              only: "set[tuple[int, str]] | None" = None) -> View:
    """The values of ``ranges`` in the workbook at ``path`` (cached values, or formulas),
    read as V2 reads them (``ws[cell_name]``, :296 and :322); ``only`` limits it to those
    ``(range, cell)`` pairs. The workbook is closed and freed before this returns."""
    wb = _load(path, data_only)
    try:
        view: View = []
        for index, (sheet, cells) in enumerate(ranges):
            ws = _find_sheet(wb, sheet)
            if ws is None:
                view.append(None)
                continue
            view.append({name: ws[name].value for name in cells
                         if only is None or (index, name) in only})
        return view
    finally:
        wb.close()
        del wb
        gc.collect()


# ── V2's comparison, on the values ────────────────────────────────────────────

def _text(v):
    """evaluation.py:333-337: an array formula's text for the error message."""
    return v.text if hasattr(v, 'text') else v


@dataclass
class Grade:
    """``compare_workbooks_with_regression`` (:347-405) and ``process_single_item``'s
    score (:452-458) for one output."""

    #: Every answer cell matched (``is_correct``, :403).
    exact: bool
    #: The error V2 reports: the first mismatch's message, "" when exact (:404-405).
    error_message: str
    regression: dict[str, int]
    modification: dict[str, int]
    #: V2's ``regression_accuracy`` and ``modification_accuracy`` (:453-457).
    regression_accuracy: float
    modification_accuracy: float
    #: V2's ``accuracy``: 1.0 or 0.0 (:458).
    accuracy: float
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.accuracy == 1.0


def score(regression: dict[str, int], modification: dict[str, int]) -> tuple[float, float, float]:
    """evaluation.py:453-458: the two ratios rounded to four places, regression at or above
    99.8% counted as 1.0, and accuracy 1.0 only when both are 1.0 (a part with no cells is
    0.0, so it fails)."""
    reg_ratio = round(regression['correct'] / regression['total'], 4) if regression['total'] else 0.0
    mod_ratio = round(modification['correct'] / modification['total'], 4) if modification['total'] else 0.0
    if reg_ratio >= 0.998:
        reg_ratio = 1.0
    accuracy = 1.0 if reg_ratio == 1.0 and mod_ratio == 1.0 else 0.0
    return reg_ratio, mod_ratio, accuracy


def _error_cells(*views: View) -> set[tuple[int, str]]:
    """The ``(range, cell)`` pairs where any view holds an Excel error value: the only
    cells V2 compares by formula (:298, :324)."""
    found: set[tuple[int, str]] = set()
    for view in views:
        for index, values in enumerate(view):
            for name, value in (values or {}).items():
                if _has_excel_error(value):
                    found.add((index, name))
    return found


def compare(ranges: list[Range], inp: View, gold: View, out: View,
            inp_f: View | None, gold_f: View | None, out_f: View | None) -> Grade:
    """``classify_cells_by_modification`` (:282-304) then
    ``cell_level_compare_with_classification`` (:307-344) for each range, summed as
    ``compare_workbooks_with_regression`` sums them (:375-405). The ``_f`` views are the
    formula views (``None`` entries where the workbook lacks the sheet)."""
    reg_stats, mod_stats = {'correct': 0, 'total': 0}, {'correct': 0, 'total': 0}
    total_correct, total_cells, error_msgs = 0, 0, []
    for index, (sheet_name, cells) in enumerate(ranges):
        ws_input, ws_answer = inp[index], gold[index]
        # :287-288
        if ws_input is None or ws_answer is None:
            regression_cells, modification_cells = [], []
        else:
            ws_input_f = inp_f[index] if inp_f is not None else None
            ws_answer_f = gold_f[index] if gold_f is not None else None
            regression_cells, modification_cells = [], []
            for cell_name in cells:
                v_in, v_ans = ws_input[cell_name], ws_answer[cell_name]
                if ws_input_f is not None and ws_answer_f is not None \
                        and (_has_excel_error(v_in) or _has_excel_error(v_ans)):
                    is_same = compare_cell_formula(ws_input_f[cell_name], ws_answer_f[cell_name])
                else:
                    is_same = compare_cell_value(v_in, v_ans)
                (regression_cells if is_same else modification_cells).append(cell_name)
        # :310-311
        if out[index] is None:
            reg_correct, reg_total = 0, len(regression_cells)
            mod_correct, mod_total = 0, len(modification_cells)
            msgs = [f"{sheet_name} worksheet not found"]
        else:
            ws_answer_f = gold_f[index] if gold_f is not None else None
            ws_output_f = out_f[index] if out_f is not None else None

            def count_correct(names, label):
                correct, errors = 0, []
                for name in names:
                    ans_val, out_val = ws_answer[name], out[index][name]
                    if ws_answer_f is not None and ws_output_f is not None \
                            and (_has_excel_error(ans_val) or _has_excel_error(out_val)):
                        matched = compare_cell_formula(ws_answer_f[name], ws_output_f[name])
                    else:
                        matched = compare_cell_value(ans_val, out_val)
                    if matched:
                        correct += 1
                    else:
                        errors.append(f"{label} error at {sheet_name}!{name}: "
                                      f"answer={_text(ans_val)}, output={_text(out_val)}")
                return correct, errors

            reg_correct, reg_errors = count_correct(regression_cells, "Regression")
            mod_correct, mod_errors = count_correct(modification_cells, "Modification")
            reg_total, mod_total = len(regression_cells), len(modification_cells)
            msgs = reg_errors + mod_errors
        reg_stats['correct'] += reg_correct
        reg_stats['total'] += reg_total
        mod_stats['correct'] += mod_correct
        mod_stats['total'] += mod_total
        total_correct += reg_correct + mod_correct
        total_cells += reg_total + mod_total
        error_msgs.extend(msgs)
    is_correct = total_correct == total_cells
    first_error = next((m for m in error_msgs if m), "")
    reg_ratio, mod_ratio, accuracy = score(reg_stats, mod_stats)
    return Grade(exact=is_correct, error_message="" if is_correct else first_error,
                 regression=reg_stats, modification=mod_stats, regression_accuracy=reg_ratio,
                 modification_accuracy=mod_ratio, accuracy=accuracy, errors=error_msgs)


class OutputUnreadable(Exception):
    """The output the model's code wrote cannot be opened: the model's failure."""


def _read(path: str, ranges: list[Range], role: str, *, data_only: bool,
          only: "set[tuple[int, str]] | None" = None) -> View:
    try:
        return read_view(path, ranges, data_only=data_only, only=only)
    except MemoryError as exc:
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the grader ran out of memory reading the "
                           f"{role} {path}: {exc!r}") from exc
    except Exception as exc:  # noqa: BLE001 - which workbook decides whose failure it is
        if role == "output":
            raise OutputUnreadable(f"{type(exc).__name__}: {exc}") from exc
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the grader could not read the {role} "
                           f"{path}: {type(exc).__name__}: {exc}") from exc


def needs_mode(input_name: str) -> str:
    """evaluation.py:431-436: the comparison mode V2 picks for a Debugging input's path:
    ``font_color`` for "Color", ``formula`` for "Embedded", else ``""``."""
    if 'Embedded' in input_name:
        return "formula"
    if 'Color' in input_name:
        return "font_color"
    return ""


def grade(input_file: str, gt_file: str, proc_file: str, answer_position: str) -> Grade:
    """V2's grade of ``proc_file`` (``compare_workbooks_with_regression`` with both modes
    off, :347-405, and ``process_single_item``'s score, :452-458). A missing output scores
    0 with V2's message (:354-355); an output that cannot be opened raises
    `OutputUnreadable`; a missing or unreadable input or golden, or memory running out,
    raises `GraderFailed`."""
    empty = {'correct': 0, 'total': 0}
    if not os.path.exists(proc_file):
        return Grade(exact=False, error_message="output file not exist", regression=dict(empty),
                     modification=dict(empty), regression_accuracy=0.0,
                     modification_accuracy=0.0, accuracy=0.0)
    for role, path in (("input", input_file), ("golden", gt_file)):
        if not os.path.exists(path):
            raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the prepared {role} {path} does not exist")
    try:
        first = _first_sheet(gt_file)
    except MemoryError as exc:
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the grader ran out of memory: {exc!r}") from exc
    except Exception as exc:  # noqa: BLE001
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the grader could not read the golden "
                           f"{gt_file}: {type(exc).__name__}: {exc}") from exc
    try:
        ranges = answer_ranges(answer_position, first)
    except Exception as exc:  # noqa: BLE001 - the dataset's position, not the model's
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the answer position {answer_position[:200]!r} "
                           f"cannot be read: {type(exc).__name__}: {exc}") from exc
    inp = _read(input_file, ranges, "input", data_only=True)
    gold = _read(gt_file, ranges, "golden", data_only=True)
    out = _read(proc_file, ranges, "output", data_only=True)
    errors = _error_cells(inp, gold, out)
    inp_f = gold_f = out_f = None
    if errors:
        inp_f = _read(input_file, ranges, "input", data_only=False, only=errors)
        gold_f = _read(gt_file, ranges, "golden", data_only=False, only=errors)
        out_f = _read(proc_file, ranges, "output", data_only=False, only=errors)
    try:
        return compare(ranges, inp, gold, out, inp_f, gold_f, out_f)
    except MemoryError as exc:
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the grader ran out of memory comparing: "
                           f"{exc!r}") from exc
    except Exception as exc:  # noqa: BLE001 - V2's except at :446-448 made this a silent 0
        raise GraderFailed(f"{HARNESS_INCIDENT_TEXT}the grader failed comparing {proc_file}: "
                           f"{type(exc).__name__}: {exc}") from exc


def feedback(g: Grade, *, max_errors: int = 20) -> str:
    """What the analysts are shown of a grade: V2's numbers and its mismatch messages
    (``error_message``, :338), the first ``max_errors`` of them."""
    lines = [f"## Eval Result: {'PASS' if g.passed else 'FAIL'}",
             f"regression (unchanged cells): {g.regression['correct']}/{g.regression['total']} "
             f"(accuracy {g.regression_accuracy}; 99.8% counts as 1.0)",
             f"modification (changed cells): {g.modification['correct']}/{g.modification['total']} "
             f"(accuracy {g.modification_accuracy})"]
    shown: Iterable[str] = g.errors[:max_errors] if g.errors else \
        ([g.error_message] if g.error_message else [])
    lines.extend(shown)
    if len(g.errors) > max_errors:
        lines.append(f"... {len(g.errors) - max_errors} more mismatches")
    return "\n".join(lines)
