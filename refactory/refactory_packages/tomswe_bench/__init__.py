"""TomSWE: preference-aware task benchmark with user profiles."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import TomSWE

PACKAGE = Package(
    name="tomswe_bench",
    description="TomSWE — preference-aware task pipeline with user profiles in containerized evaluation.",
    modes=(TomSWE,),
    ops=("refactory_packages.tomswe_bench.ops",),
)
