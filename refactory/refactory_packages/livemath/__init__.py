from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import Solver, SolverJudge, SolverSelfCheck, SolverVerify
from .skill_graph import SkillGraph

PACKAGE = Package(
    name="livemath",
    description="LiveMathematicianBench task with a one-node solver graph",
    modes=(Solver, SolverVerify, SolverJudge, SolverSelfCheck, SkillGraph),
    ops=("refactory_packages.livemath.ops", "refactory_packages.livemath.block_ops",
         "refactory_packages.livemath.skill_graph"),
)
