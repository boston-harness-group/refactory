from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import Solver

PACKAGE = Package(
    name="livemath",
    description="LiveMathematicianBench task with a one-node solver graph",
    modes=(Solver,),
    ops=("refactory_packages.livemath.ops",),
)
