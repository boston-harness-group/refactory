"""GEPA outer-loop strategy: a port of ``gepa.optimize`` (gepa-ai/gepa 0.1.4, MIT).

Reflective evolution of instruction knobs with a per-instance Pareto frontier
(Agrawal et al., 2025, "GEPA: Reflective Prompt Evolution Can Outperform
Reinforcement Learning"). See ``refactory_packages/gepa/loop.py``.
"""

from refactory_dsh.nn import Package

from refactory_packages.gepa.loop import GepaLoop

PACKAGE = Package(
    name="gepa",
    description="GEPA outer-loop strategy (gepa-ai/gepa, MIT).",
    modes=(),
    strategies=(GepaLoop,),
)
