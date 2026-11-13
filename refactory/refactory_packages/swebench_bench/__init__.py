"""SWE-bench benchmark: minimal bug-fix pipeline for containerized evaluation."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import SWEBench

PACKAGE = Package(
    name="swebench_bench",
    description="SWE-bench benchmark — 4-node pipeline for solving GitHub issues in containerized evaluation.",
    modes=(SWEBench,),
    ops=("refactory_packages.swebench_bench.ops",),
)
