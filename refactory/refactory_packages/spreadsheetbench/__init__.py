from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import Solver

PACKAGE = Package(
    name="spreadsheetbench",
    description="SpreadsheetBench Verified 400 with SkillOpt's multi-turn codegen solver",
    modes=(Solver,),
    ops=("refactory_packages.spreadsheetbench.ops", "refactory_dsh.feedback"),
)
