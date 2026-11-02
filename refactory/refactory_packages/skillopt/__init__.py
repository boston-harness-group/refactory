"""SkillOpt replay package — offline replay of arm A's recorded runs through arm B.

This is a utility package with no modes: the replay model client,
the comparison report, and the runner live here so they can be imported
from ``refactory_packages.skillopt.*`` without polluting the mode registry.
"""

from refactory_dsh.nn import Package

PACKAGE = Package(
    name="skillopt",
    description="Offline replay infrastructure for SkillOpt arm-B testing.",
)
