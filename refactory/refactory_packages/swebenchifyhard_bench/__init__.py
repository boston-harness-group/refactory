"""SWE-bench Verify Hard: synthetic Go bug-fix benchmark."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import SWEBenchifyHard

PACKAGE = Package(
    name="swebenchifyhard_bench",
    description="SWE-bench Verify Hard — 284 synthetic Go bug-fix instances in containerized evaluation.",
    modes=(SWEBenchifyHard,),
    ops=("refactory_packages.swebenchifyhard_bench.ops",),
)
