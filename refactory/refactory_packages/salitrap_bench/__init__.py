"""Salitrap: commonsense reasoning benchmark under salience bias."""

from __future__ import annotations

from refactory_dsh.nn import Package

from .mode import Salitrap

PACKAGE = Package(
    name="salitrap_bench",
    description="Salitrap — commonsense reasoning pipeline under salience bias in containerized evaluation.",
    modes=(Salitrap,),
    ops=("refactory_packages.salitrap_bench.ops",),
)
