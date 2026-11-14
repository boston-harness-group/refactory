"""SkillOpt outer-loop strategy for refactory.

Reproduces the SkillOpt algorithm (microsoft/SkillOpt, MIT) as an OuterLoop
subclass built entirely from refactory's public extension points.

The package contributes the strategy through ``Package.strategies``, so
``strategies.resolve("skillopt")`` (and ``--strategy skillopt`` on the CLI)
finds it by package discovery. It is not added to the library's own
``STRATEGIES`` table, which counts the SRF ports.

This is a *strategy* package, not a mode package: it declares no modes,
only a strategy that drives any mode with a text knob (``skill`` by default, the
``knob`` option) or several (the ``knobs`` option, seen as one document of
sections), and ``SkillOptProposer`` (``proposer.py``), its optimizer steps
as a part another strategy can call.
"""

from refactory_dsh.nn import Package

from refactory_packages.skillopt.loop import SkillOptLoop

PACKAGE = Package(
    name="skillopt",
    description="SkillOpt outer-loop strategy (microsoft/SkillOpt, MIT).",
    modes=(),
    strategies=(SkillOptLoop,),
)
