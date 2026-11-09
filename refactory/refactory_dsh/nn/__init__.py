"""The mode library: the ``nn`` of the harness world.

A mode is a Python class. Its architecture and parameters are a serializable
state dict, its score and its invariants are methods no optimizer can reach, and
every structural change is typed data validated before it costs anything.

    from refactory_dsh import nn

    class MyMode(nn.Mode):
        @classmethod
        def declare(cls):
            return {"rounds": nn.Knob(3, "int", lo=1, hi=8)}

        def build(self):
            graph = nn.Graph()
            graph.add(nn.Node("run", "op", call="mypkg.ops:run", knobs=("rounds",)))
            return graph

        def score(self, run):
            return float(run.facts["best"])
"""

from __future__ import annotations

from . import edits, resolve
from .declarations import DeclarationError, Memory, Port
from .executor import (ExecutionError, Executor, ModelClient, RunContext, Step, Verdict, execute,
                       fills_facts)
from .graph import Edge, Graph, GraphError, Node
from .knob import KINDS, Knob, KnobError
from .mode import STATE_VERSION, Mode, RunResult, StateError
from .package import Package, PackageError, mode_id, split
from .validate import (
    RULES,
    Invariant,
    Issue,
    declared_knobs,
    fact_produced,
    artifact_produced,
    on_path_nodes,
    produced_on_path,
    rules_text,
    validate,
    validate_state,
)

__all__ = [
    "KINDS",
    "STATE_VERSION",
    "DeclarationError",
    "Edge",
    "ExecutionError",
    "Executor",
    "Graph",
    "GraphError",
    "Invariant",
    "Issue",
    "Knob",
    "KnobError",
    "Memory",
    "Mode",
    "ModelClient",
    "Node",
    "RULES",
    "Package",
    "Port",
    "PackageError",
    "RunContext",
    "RunResult",
    "StateError",
    "Step",
    "Verdict",
    "declared_knobs",
    "edits",
    "execute",
    "fact_produced",
    "fills_facts",
    "artifact_produced",
    "mode_id",
    "on_path_nodes",
    "produced_on_path",
    "rules_text",
    "resolve",
    "split",
    "validate",
    "validate_state",
]
