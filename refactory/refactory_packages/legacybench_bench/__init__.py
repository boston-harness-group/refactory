"""LegacyBench: legacy code bug-fix benchmark (COBOL, Fortran, C, Java 7, Assembly)."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import LegacyBench

PACKAGE = Package(
    name="legacybench_bench",
    description="LegacyBench — legacy code bug-fix pipeline for multi-language containerized evaluation.",
    modes=(LegacyBench,),
    ops=("refactory_packages.legacybench_bench.ops",),
)
