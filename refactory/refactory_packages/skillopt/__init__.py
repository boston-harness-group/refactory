"""SkillOpt outer-loop strategy and replay package for refactory.

Reproduces the SkillOpt algorithm (microsoft/SkillOpt, MIT) as an OuterLoop
subclass built entirely from refactory's public extension points, and provides
offline replay infrastructure for arm-B testing.

The package contributes the strategy through ``Package.strategies``, so
``strategies.resolve("skillopt")`` (and ``--strategy skillopt`` on the CLI)
finds it by package discovery.  It also declares a Replay mode for offline
replay testing whose graph validates, satisfying the registry invariant.
"""

from __future__ import annotations

from refactory_dsh import nn
from refactory_dsh.nn import Package

from refactory_packages.skillopt.loop import SkillOptLoop

OPS = "refactory_packages.skillopt.ops"


class Replay(nn.Mode):
    """Offline replay mode for SkillOpt arm-B testing.

    Unscored: the replay is judged by the comparison report against arm A,
    not by a score function.
    """

    name = "replay"
    description = "offline replay test for SkillOpt arm-B divergence detection"
    unscored_reason = (
        "offline replay: divergences are detected by comparing B's behaviour "
        "against A's recorded outputs, not by a score function"
    )

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "recording_dir": nn.Knob(
                "", "str",
                description="path to arm A's recording directory",
            ),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "check", "op",
            call=f"{OPS}:replay_noop",
            knobs=("recording_dir",),
            produces=("replay_ready",),
        ))
        return graph


PACKAGE = Package(
    name="skillopt",
    description="SkillOpt outer-loop strategy and offline replay infrastructure.",
    modes=(Replay,),
    ops=(OPS,),
    strategies=(SkillOptLoop,),
)


def register() -> None:
    """Register the SkillOpt strategy in ``STRATEGIES`` by hand (optional)."""
    from refactory_dsh import strategies
    strategies.register("skillopt", SkillOptLoop)
