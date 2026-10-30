"""DevOpsGym: build and configuration task benchmark."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import DevOpsGym

PACKAGE = Package(
    name="devopsgym_bench",
    description="DevOpsGym — build/configuration task pipeline for containerized evaluation.",
    modes=(DevOpsGym,),
    ops=("refactory_packages.devopsgym_bench.ops",),
)
