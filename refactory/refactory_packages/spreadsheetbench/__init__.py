from __future__ import annotations

from refactory_dsh.nn import Package

from .agent_mode import Agent
from .mode import Solver

PACKAGE = Package(
    name="spreadsheetbench",
    description=("SpreadsheetBench Verified 400 with SkillOpt's multi-turn codegen solver and "
                 "its Claude Code agent harness"),
    modes=(Solver, Agent),
    ops=("refactory_packages.spreadsheetbench.ops", "refactory_packages.spreadsheetbench.agent_ops",
         "refactory_dsh.feedback"),
)
