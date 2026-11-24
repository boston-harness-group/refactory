"""A Task that lays an item file into each pass, and ops and modes that read it, for
`test_structure_reach` (the spine's node commands import this module through PYTHONPATH).

Each item's input is ``INPUT-<id>``; its grader-only answer, ``GOLD-<id>-SECRET``, lives in
the item's metadata and, for `GoldLinkTask`, in a file outside the pass. A pass scores 1
when its ``answer.txt`` holds the item's own input.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from refactory_dsh import nn
from refactory_dsh.task import Task, TaskInstance, ToolEnvironment, VerifyResult

MODULE = "item_files_ops"
#: The item file every item gets, under the same name.
INPUT = "item/input.bin"
#: What a grader-only answer starts with, so a leak is found by content.
SECRET = "SECRET"
ANSWER = "answer.txt"


def item_input(item_id: str) -> str:
    return f"INPUT-{item_id}"


def gold(item_id: str) -> str:
    return f"GOLD-{item_id}-{SECRET}"


class FileTask(Task):
    """Two items, each with an input laid into its pass as ``item/input.bin``."""

    name = "item_files"
    item_files = MappingProxyType({"input": INPUT})
    grader_fields = ("gold",)
    graded_files = (ANSWER,)

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[TaskInstance]:
        return [TaskInstance(id=item, metadata={"gold": gold(item)}) for item in ("a", "b")]

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        target = Path(workspace) / INPUT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item_input(instance.id).encode())

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        answer = Path(workspace) / ANSWER
        text = answer.read_text() if answer.exists() else ""
        passed = item_input(instance.id) in text and SECRET not in text
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0, details=text[:200])


class GoldLinkTask(FileTask):
    """A Task whose ``setup`` lays the item file as a link to the answer kept outside the
    pass: the runtime must refuse it, so no step reads the answer through an item file."""

    name = "item_files_gold_link"

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        outside = Path(workspace).parent / f"gold-{instance.id}.txt"
        outside.write_text(gold(instance.id))
        target = Path(workspace) / INPUT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(outside)


def read_input(ctx: Any) -> dict[str, Any]:
    """Copy the item's input, opened through `RunContext.item_file`, into the answer."""
    path = ctx.item_file("input")
    text = path.read_text() if path is not None else ""
    ctx.write_text(ANSWER, text)
    return {"echo": text}


def peek_input(ctx: Any) -> dict[str, Any]:
    """Read the item's input by name through the context's read helper, not `item_file`."""
    text = ctx.read_text(INPUT)
    ctx.write_text(ANSWER, text)
    return {"echo": text}


class GoldHardLinkTask(FileTask):
    """A Task whose ``setup`` lays the item file as a hard link to the answer kept outside
    the pass: refused like a symbolic one."""

    name = "item_files_gold_hard_link"

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        import os

        outside = Path(workspace).parent / f"gold-{instance.id}.txt"
        outside.write_text(gold(instance.id))
        target = Path(workspace) / INPUT
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(outside, target)


class ItemFilesMode(nn.Mode):
    """A data pass whose one op reads the item file: ``reads`` says whether the op
    declares it (`RunContext.item_file` refuses it when it does not)."""

    name = "item_files_probe"
    task_ref = f"{MODULE}:FileTask"
    reads: tuple[str, ...] = (INPUT,)
    data_writes: tuple[str, ...] = ("current_item.json", INPUT)
    call = f"{MODULE}:read_input"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"data_parallelism": nn.Knob(1, "int", lo=1, hi=4, frozen=True,
                                            description="items at once")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="read", subgraph_exit="read",
                          task_ref=self.task_ref, writes=self.data_writes,
                          produces=("item_scores",), parallelism_knob="data_parallelism",
                          knobs=("data_parallelism",)))
        graph.add(nn.Node("read", "op", call=self.call, reads=self.reads,
                          writes=(ANSWER,), produces=("echo",)))
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [r["score"] for r in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class UndeclaredRead(ItemFilesMode):
    """The op opens the item file without declaring it."""

    name = "item_files_undeclared"
    reads = ()


class UndeclaredReadText(ItemFilesMode):
    """The op reads the item file by name through ``ctx.read_text`` without declaring it."""

    name = "item_files_undeclared_read_text"
    reads = ()
    call = f"{MODULE}:peek_input"


class DeclaredReadText(UndeclaredReadText):
    """The same op, declaring the item file: allowed."""

    name = "item_files_declared_read_text"
    reads = (INPUT,)


class GoldHardLink(ItemFilesMode):
    """The Task lays its item file as a hard link to the answer."""

    name = "item_files_gold_hard_link"
    task_ref = f"{MODULE}:GoldHardLinkTask"


class GoldLink(ItemFilesMode):
    """The Task lays its item file as a link to the answer."""

    name = "item_files_gold_link"
    task_ref = f"{MODULE}:GoldLinkTask"


class UndeclaredWrite(ItemFilesMode):
    """The data node leaves the Task's item file out of its writes (V12)."""

    name = "item_files_unwritten"
    data_writes = ("current_item.json",)


class SealedLook(ItemFilesMode):
    """A sealed agent step reads the item file in its own workspace (`workspace._seal`)."""

    name = "item_files_sealed"

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="look", subgraph_exit="look",
                          task_ref=self.task_ref, writes=self.data_writes,
                          produces=("item_scores",), parallelism_knob="data_parallelism",
                          knobs=("data_parallelism",)))
        graph.add(nn.Node("look", "agent", prompt_template="Look at the item's input.",
                          reads=(INPUT,), writes=(ANSWER,), tools=("bash",),
                          workspace="scratch"))
        return graph


#: Names another interpreter for `ToolEnvTask` (a test's own virtual environment).
PYTHON_ENV = "ITEM_FILES_TEST_PYTHON"


def test_interpreter() -> str:
    """The interpreter a `ToolEnvTask`'s sealed steps run as ``python3``: `PYTHON_ENV`, else
    the one running the tests."""
    import os
    import sys

    return os.environ.get(PYTHON_ENV) or sys.executable


#: What `ToolEnvTask` says of its tool environment.
TOOL_SUMMARY = "python3 (and python) = the tests' own interpreter, with pytest"


class ToolEnvTask(FileTask):
    """`FileTask` whose sealed tool steps run its interpreter (`Task.tool_environment`)."""

    name = "item_files_tool_env"
    tool_environment = ToolEnvironment(summary=TOOL_SUMMARY, python=test_interpreter)


#: Names the grader-side path `HiddenEnvTask` declares (`ToolEnvironment.hidden`).
HIDDEN_ENV = "ITEM_FILES_TEST_HIDDEN"


def test_hidden(workdir: Path) -> list[str]:
    import os

    return [os.environ[HIDDEN_ENV]] if os.environ.get(HIDDEN_ENV) else []


class HiddenEnvTask(FileTask):
    """`ToolEnvTask` whose environment names a grader-side path no folder may show."""

    name = "item_files_hidden_env"
    tool_environment = ToolEnvironment(summary=TOOL_SUMMARY, python=test_interpreter,
                                       hidden=test_hidden)


def missing_interpreter() -> str:
    raise RuntimeError("the pinned interpreter is missing (build it first)")


class MissingEnvTask(FileTask):
    """`FileTask` whose declared interpreter cannot be had: its sealed steps are refused."""

    name = "item_files_missing_env"
    tool_environment = ToolEnvironment(summary=TOOL_SUMMARY, python=missing_interpreter)


class SealedEnvLook(SealedLook):
    """`SealedLook` whose Task declares a tool environment."""

    name = "item_files_sealed_env"
    task_ref = f"{MODULE}:ToolEnvTask"


class SealedMissingEnv(SealedLook):
    """`SealedLook` whose Task's tool environment cannot be had."""

    name = "item_files_sealed_missing_env"
    task_ref = f"{MODULE}:MissingEnvTask"


class WorkbookTask(ToolEnvTask):
    """`ToolEnvTask` whose setup creates an xlsx workbook for each item (A1, B1 values)."""

    name = "item_files_workbook"
    item_files = MappingProxyType({"input": INPUT, "workbook": "workbook.xlsx"})

    def setup(self, instance: TaskInstance, workspace: Path, **_kw: Any) -> None:
        super().setup(instance, workspace)
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = 10
        ws["B1"] = 5
        wb.save(Path(workspace) / "workbook.xlsx")


def _find_libreoffice() -> str | None:
    """The soffice binary's program directory, if one is installed."""
    import os
    import shutil

    for cmd in ("soffice", "libreoffice"):
        path = shutil.which(cmd)
        if path:
            return os.path.realpath(os.path.dirname(os.path.realpath(path)))
    return None


def _lo_commands() -> dict[str, str]:
    prog = _find_libreoffice()
    if not prog:
        raise RuntimeError("LibreOffice not found")
    soffice_bin = os.path.join(prog, "soffice.bin")
    if not os.path.isfile(soffice_bin):
        soffice_bin = os.path.join(prog, "soffice")
    return {"soffice": (f'#!/bin/sh\n'
                        f'export SAL_USE_VCLPLUGIN=svp\n'
                        f'export LD_LIBRARY_PATH="{prog}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n'
                        f'export SAL_TMPDIR=/tmp\n'
                        f'export TMPDIR=/tmp\n'
                        f'exec "{soffice_bin}" "$@"\n')}


def _lo_reads() -> list[str]:
    prog = _find_libreoffice()
    if not prog:
        raise RuntimeError("LibreOffice not found")
    lo_root = os.path.dirname(prog)
    return [prog, os.path.join(lo_root, "share")] if os.path.isdir(os.path.join(lo_root, "share")) else [prog]


LO_TOOL_SUMMARY = "python3 + soffice (LibreOffice headless)"


class LibreOfficeTask(FileTask):
    """`FileTask` whose sealed steps get the test interpreter and soffice."""

    name = "item_files_libreoffice"
    item_files = MappingProxyType({"input": INPUT, "workbook": "workbook.xlsx"})
    tool_environment = ToolEnvironment(
        summary=LO_TOOL_SUMMARY, python=test_interpreter,
        commands=_lo_commands, reads=_lo_reads, private_tmp=True,
    )

    def setup(self, instance: TaskInstance, workspace: Path, **_kw: Any) -> None:
        super().setup(instance, workspace)
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = 10
        ws["B1"] = 20
        ws["C1"] = "=A1+B1"
        wb.save(Path(workspace) / "workbook.xlsx")


#: Code for a sealed op that reads a workbook with openpyxl and returns the total.
WORKBOOK_READ_CODE = (
    "import openpyxl\n"
    "def main(inp):\n"
    "    wb = openpyxl.load_workbook('workbook.xlsx')\n"
    "    ws = wb.active\n"
    "    total = ws['A1'].value + ws['B1'].value\n"
    "    return {'facts': {'total': str(total)},\n"
    "            'files': {'result.txt': str(total)}}\n"
)

#: Code for a sealed gate that reads a workbook, checks if the answer mentions the total.
WORKBOOK_GATE_CODE = (
    "import openpyxl\n"
    "def main(inp):\n"
    "    wb = openpyxl.load_workbook('workbook.xlsx')\n"
    "    ws = wb.active\n"
    "    total = ws['A1'].value + ws['B1'].value\n"
    "    answer = open('answer.txt').read()\n"
    "    if str(total) in answer:\n"
    "        return {'verdict': 'proceed', 'reason': f'answer has {total}'}\n"
    "    return {'verdict': 'reloop', 'reason': f'say {total}'}\n"
)


class SealedWorkbookRead(ItemFilesMode):
    """Agent writes an answer, then a sealed code op reads the workbook with openpyxl."""

    name = "sealed_workbook_read"
    task_ref = f"{MODULE}:WorkbookTask"
    data_writes: tuple[str, ...] = ("current_item.json", INPUT, "workbook.xlsx")

    def build(self) -> nn.Graph:
        from refactory_dsh.snippet import RUN

        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit="read_wb",
                          task_ref=self.task_ref, writes=self.data_writes,
                          produces=("item_scores", "total"),
                          parallelism_knob="data_parallelism",
                          knobs=("data_parallelism",)))
        graph.add(nn.Node("solve", "agent", prompt_template="Write your answer.",
                          writes=(ANSWER,), tool_policy="none"))
        graph.add(nn.Node("read_wb", "op", call=RUN, sealed_script=True,
                          reads=("workbook.xlsx",), required_reads=("workbook.xlsx",),
                          produces=("total",), writes=("result.txt",),
                          code=WORKBOOK_READ_CODE))
        graph.connect("solve", "read_wb")
        return graph


class SealedWorkbookLoop(ItemFilesMode):
    """Agent writes an answer, a sealed code gate reads the workbook and loops."""

    name = "sealed_workbook_loop"
    task_ref = f"{MODULE}:WorkbookTask"
    data_writes: tuple[str, ...] = ("current_item.json", INPUT, "workbook.xlsx")

    def build(self) -> nn.Graph:
        from refactory_dsh.nn.loops import LOOP_END_CALL
        from refactory_dsh.snippet import GATE

        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit="done",
                          task_ref=self.task_ref, writes=self.data_writes,
                          produces=("item_scores",), parallelism_knob="data_parallelism",
                          knobs=("data_parallelism",)))
        graph.add(nn.Node("solve", "agent", prompt_template="Write your answer.",
                          writes=(ANSWER,), tool_policy="none"))
        graph.add(nn.Node("check_wb", "gate", evaluator=GATE, sealed_script=True,
                          reads=("workbook.xlsx", ANSWER),
                          required_reads=("workbook.xlsx", ANSWER),
                          max_iterations=1, code=WORKBOOK_GATE_CODE))
        graph.add(nn.Node("done", "op", call=LOOP_END_CALL))
        graph.connect("solve", "check_wb")
        graph.connect("check_wb", "solve", condition="reloop")
        graph.connect("check_wb", "done", condition="proceed")
        return graph


class AgentPass(ItemFilesMode):
    """A data pass whose one agent step (no tools) answers, then a step after the pass:
    where the outer loop's operators add steps, inside the per-item part and outside it."""

    name = "item_files_agent_pass"
    task_ref = f"{MODULE}:ToolEnvTask"

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit="solve",
                          task_ref=self.task_ref, writes=self.data_writes,
                          produces=("item_scores",), parallelism_knob="data_parallelism",
                          knobs=("data_parallelism",)))
        graph.add(nn.Node("solve", "agent", prompt_template="Answer the item.",
                          writes=(ANSWER,), tool_policy="none"))
        graph.add(nn.Node("report", "agent", prompt_template="Report on the run.",
                          writes=("report.txt",), tool_policy="none"))
        graph.connect("data", "report")
        return graph


def records(facts_path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(facts_path).read_text())["item_scores"]
