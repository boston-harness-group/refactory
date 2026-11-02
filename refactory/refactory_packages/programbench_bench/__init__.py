"""ProgramBench: adversarial binary reverse-engineering benchmark."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import ProgramBench

PACKAGE = Package(
    name="programbench_bench",
    description="ProgramBench — adversarial binary reverse-engineering pipeline in containerized evaluation.",
    modes=(ProgramBench,),
    ops=("refactory_packages.programbench_bench.ops",),
)
