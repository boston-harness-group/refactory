from __future__ import annotations

from refactory_dsh.nn import Package

from .agent_mode import Agent

PACKAGE = Package(
    name="spreadsheetbench_v2",
    description=("SpreadsheetBench 2 (277 file-graded tasks, prepared and split by workbook) "
                 "with SkillOpt's Claude Code agent harness and V2's grader"),
    modes=(Agent,),
    ops=("refactory_packages.spreadsheetbench_v2.agent_ops",
         "refactory_packages.spreadsheetbench.agent_ops",
         "refactory_packages.spreadsheetbench.ops"),
)
