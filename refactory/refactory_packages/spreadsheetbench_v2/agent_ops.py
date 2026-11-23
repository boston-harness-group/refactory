"""The ops of the SpreadsheetBench 2 agent harness: SkillOpt's, with V2's task text.

The graph is `spreadsheetbench.agent_mode.Agent`'s (``prepare -> has_case -> solve ->
finalize``) and so are its pieces (`spreadsheetbench.agent_ops`): the same workspace
files, the same sealed solve step, the same run of ``solution.py`` in the code sandbox.
What differs is the task, and where the data lies:

- `prepare` writes as ``task.md`` V2's own task block -- the ``<instructions>`` section of
  V2's SWE-agent ``instance_template`` (``SWE-agent/config/spreadsheet.yaml:49-60``): its
  instruction, the input file and the output path, which here are ``input.xlsx`` and
  ``output.xlsx``, the names SkillOpt's ``run_solution.py`` gives ``INPUT_PATH`` and
  ``OUTPUT_PATH`` -- then SkillOpt's harness section (`agent_port.CODEX_TASK_SECTION`).
  V2's tools, workflow and "Important" notes (``spreadsheet.yaml:4-47``) describe its
  SWE-agent harness (``bash``, ``view_xlsx``, ``submit``) and are not sent; this harness
  is SkillOpt's (Read, Bash, ``solution.py``).
- `finalize` runs the code blind to the prepared V2 data (`tasks.data_roots`).
"""
from __future__ import annotations

import time
from typing import Any

from refactory_packages.spreadsheetbench import agent_ops as sb_agent_ops
from refactory_packages.spreadsheetbench import agent_port, ops
from refactory_packages.spreadsheetbench import skillopt_port as port
from refactory_packages.spreadsheetbench.tasks import TURN

from . import tasks


# SWE-agent/config/spreadsheet.yaml:49-60, the instance template's instructions block,
# with ``{{spreadsheet_path}}`` and ``{{output_path}}`` the workspace's two names.
def v2_instructions(instruction: str, input_file: str = agent_port.INPUT_PATH,
                    output_path: str = "output.xlsx") -> str:
    return (
        "<instructions>\n"
        "\n"
        "## Task instructions\n"
        "You need to process a spreadsheet file based on specific instructions.\n"
        f"**Instruction:** {instruction}\n"
        "\n"
        "## Input and Output\n"
        "You are provided with an input spreadsheet file and you need to generate an output "
        "spreadsheet file based on the output_path.\n"
        f"**Input File:** {input_file}\n"
        f"**Output Path:** {output_path}\n"
        "\n"
        "</instructions>"
    )


def build_task(instruction: str) -> str:
    """task.md: V2's instructions block, then SkillOpt's harness section."""
    return f"{v2_instructions(instruction)}\n\n{agent_port.CODEX_TASK_SECTION}"


def prepare(ctx: Any) -> dict[str, Any]:
    """Record the item's start and write the workspace files (SkillOpt's, with V2's task);
    and, for the analysts, the agent's system prompt (Claude Code's own: nothing
    appended, so empty), its task and the input's preview."""
    started = time.time()
    item = ops._item(ctx)  # noqa: SLF001
    source = ops._input(ctx)  # noqa: SLF001
    if source is None:
        ctx.write_json(TURN, {"start": started, "status": "no_case"})
        return {"has_case": False}
    try:
        preview = port.preview_workbook(str(source))
    except Exception:  # noqa: BLE001 - as SkillOpt's
        preview = "(preview failed)"
    task = build_task(str(item.get("instruction", "")))
    sb_agent_ops.write_workspace(ctx, source, ctx.knob("skill", ""), task)
    ctx.write_text(ops.PREVIEW_FILE, preview)
    ctx.write_text(ops.SYSTEM_FILE, "")
    ctx.write_text(ops.USER_FILE, task)
    ctx.write_json(TURN, {"start": started, "status": ""})
    return {"has_case": True}


def finalize(ctx: Any) -> dict[str, Any]:
    """SB's agent `finalize`, the code blind to the prepared V2 data."""
    return sb_agent_ops.finalize_item(ctx, tasks.data_roots())
