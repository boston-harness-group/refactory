"""A Task that lays an item file into each pass, and ops and modes that read it, for
`test_structure_reach` (the spine's node commands import this module through PYTHONPATH).

Each item's input is ``INPUT-<id>``; its grader-only answer, ``GOLD-<id>-SECRET``, lives in
the item's metadata and, for `GoldLinkTask`, in a file outside the pass. A pass scores 1
when its ``answer.txt`` holds the item's own input.
"""
from __future__ import annotations

import json
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
