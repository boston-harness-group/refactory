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
from refactory_dsh.task import Task, TaskInstance, VerifyResult

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


class ItemFilesMode(nn.Mode):
    """A data pass whose one op reads the item file: ``reads`` says whether the op
    declares it (`RunContext.item_file` refuses it when it does not)."""

    name = "item_files_probe"
    task_ref = f"{MODULE}:FileTask"
    reads: tuple[str, ...] = (INPUT,)
    data_writes: tuple[str, ...] = ("current_item.json", INPUT)

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
        graph.add(nn.Node("read", "op", call=f"{MODULE}:read_input", reads=self.reads,
                          writes=(ANSWER,), produces=("echo",)))
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [r["score"] for r in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class UndeclaredRead(ItemFilesMode):
    """The op opens the item file without declaring it."""

    name = "item_files_undeclared"
    reads = ()


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


def records(facts_path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(facts_path).read_text())["item_scores"]
