"""SkillOpt outer-loop strategy for refactory.

Reproduces the SkillOpt algorithm (microsoft/SkillOpt, MIT) as an OuterLoop
subclass built entirely from refactory's public extension points.

Registration is explicit — call ``register()`` to add the strategy to the
registry — so that the library's SRF_PORT_COUNT invariant is never broken
by an import side-effect.

This is a *strategy* package, not a mode package: it declares no modes,
only a strategy that drives any mode with a ``skill`` knob.
"""

from refactory_dsh.nn import Package

from refactory_packages.skillopt.loop import SkillOptLoop

PACKAGE = Package(
    name="skillopt",
    description="SkillOpt outer-loop strategy (microsoft/SkillOpt, MIT).",
    modes=(),
)


def register() -> None:
    """Register the SkillOpt strategy without editing the library."""
    from refactory_dsh import strategies
    strategies.register("skillopt", SkillOptLoop)
