"""SpreadsheetBench 2 agent mode: SkillOpt's Claude Code agent harness on V2's tasks.

The graph, knobs and run settings are `spreadsheetbench.agent_mode.Agent`'s -- one sealed
Claude Code session per item with ``Read`` and ``Bash`` in a workspace holding
``input.xlsx``, the skill as ``.agents/skills/skillopt-target/SKILL.md``, ``task.md`` and
``run_solution.py``; it writes ``solution.py``, which is run on the input in the code
sandbox and graded -- over SpreadsheetBench 2's items (`tasks.SpreadsheetBenchV2Task`),
with V2's task text (`agent_ops`) and V2's grade. SkillOpt never ran V2: the harness is
SkillOpt's pattern, the task text, data and grader are V2's (docs/spreadsheetbench.md,
"SpreadsheetBench 2").

As in SB's agent mode there is no wall-clock or turn cap: the session ends on inactivity,
and the one limit is the harness backstop. No gold feedback reaches the agent: one
session has no later turn to tell (``Task.train_feedback`` is not offered).
"""
from __future__ import annotations

from refactory_dsh import nn

from refactory_packages.spreadsheetbench import agent_mode as sb_agent_mode

TASK = "refactory_packages.spreadsheetbench_v2.tasks:SpreadsheetBenchV2Task"
AGENT_OPS = "refactory_packages.spreadsheetbench_v2.agent_ops"
#: Concurrent items: fewer than SB's 24, since each item's grade recalculates a V2
#: workbook with LibreOffice and loads it (a Debugging workbook takes about 1 GB).
DATA_PARALLELISM = 8


class Agent(sb_agent_mode.Agent):
    """SkillOpt's agent harness over SpreadsheetBench 2's tasks."""

    name = "agent"
    description = ("solve a SpreadsheetBench 2 task with SkillOpt's Claude Code agent harness: "
                   "one sealed Claude Code session with Read and Bash writes solution.py")
    task_ref = TASK
    ops_module = AGENT_OPS

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        knobs = dict(super().declare())
        knobs["data_parallelism"] = nn.Knob(DATA_PARALLELISM, "int", lo=1, hi=64, frozen=True,
                                            description="concurrent items")
        return knobs
