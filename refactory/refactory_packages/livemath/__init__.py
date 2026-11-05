from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import Solver, SolverJudge, SolverSelfCheck, SolverVerify

PACKAGE = Package(
    name="livemath",
    description="LiveMathematicianBench task with a one-node solver graph",
    modes=(Solver, SolverVerify, SolverJudge, SolverSelfCheck),
    ops=("refactory_packages.livemath.ops", "refactory_packages.livemath.block_ops"),
)
