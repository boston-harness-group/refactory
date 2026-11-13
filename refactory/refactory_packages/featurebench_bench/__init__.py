"""FeatureBench: feature implementation benchmark with interface specs."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import FeatureBench

PACKAGE = Package(
    name="featurebench_bench",
    description="FeatureBench — feature implementation pipeline with interface specs in containerized evaluation.",
    modes=(FeatureBench,),
    ops=("refactory_packages.featurebench_bench.ops",),
)
