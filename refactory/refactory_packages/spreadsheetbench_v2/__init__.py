from __future__ import annotations

from refactory_dsh.nn import Package

from .agent_mode import Agent
from .steps_mode import Steps
from .swe_mode import Swe

PACKAGE = Package(
    name="spreadsheetbench_v2",
    description=("SpreadsheetBench 2 (277 file-graded tasks, prepared and split by workbook) "
                 "with SkillOpt's Claude Code agent harness or V2's own agent setup (whole, or "
                 "split into its workflow's steps), and V2's grader"),
    modes=(Agent, Swe, Steps),
    ops=("refactory_packages.spreadsheetbench_v2.agent_ops",
         "refactory_packages.spreadsheetbench_v2.swe_ops",
         "refactory_packages.spreadsheetbench_v2.steps_ops",
         "refactory_packages.spreadsheetbench.agent_ops",
         "refactory_packages.spreadsheetbench.ops"),
)
