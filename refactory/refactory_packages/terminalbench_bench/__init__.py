"""TerminalBench: real-world engineering task benchmark."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import TerminalBench

PACKAGE = Package(
    name="terminalbench_bench",
    description="TerminalBench — real-world engineering tasks in containerized terminal evaluation.",
    modes=(TerminalBench,),
    ops=("refactory_packages.terminalbench_bench.ops",),
)
