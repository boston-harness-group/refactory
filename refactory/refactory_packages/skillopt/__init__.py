"""SkillOpt replay package — offline replay of arm A's recorded runs through arm B.

Provides the replay model client, comparison report, and a replay mode whose
graph validates, so the package is discoverable and satisfies the package
registry's invariant that every installed package declares at least one mode.
"""

from __future__ import annotations

from refactory_dsh import nn

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


PACKAGE = nn.Package(
    name="skillopt",
    description="Offline replay infrastructure for SkillOpt arm-B testing.",
    modes=(Replay,),
    ops=(OPS,),
)
