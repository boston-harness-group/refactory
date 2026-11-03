"""LiveMathematicianBench one-node solver mode.

The solver graph: data → [format_prompt → solve] → finish.

System prompt ported from SkillOpt (MIT) —
skillopt/envs/livemathematicianbench/prompts/rollout_system.md.
SkillOpt solver defaults:
  exec_timeout = 1800 s -- applied as the solve node's ``timeout_s``
    (``Solver.solve_timeout_s``). SkillOpt's LiveMath config says 300
    (configs/livemathematicianbench/default.yaml:17), but arm A's runs override it
    with ``--cfg-options env.exec_timeout=1800`` (SkillOpt's run-train.sh:14 and
    run-train-seed.sh:20, recorded in each run's out/config.json ``exec_timeout``),
    and the outer-loop protocol has arm B match arm A.
  max_completion_tokens = 16384 (livemathematicianbench/default.yaml:16) -- the
    ``max_completion_tokens`` knob (``0`` = no cap), applied to the solve node's
    route only (``Solver.run``).
  reasoning_effort = medium (_base_/default.yaml:10) -- an agent node carries no
    per-node effort; it is a run setting (``--effort medium``, or ``node_routes``
    for ``solve``), not part of this graph.
  model = gpt-5.5 (_base_/default.yaml:7) -- likewise a run setting.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar, Mapping

from refactory_dsh import nn

OPS = "refactory_packages.livemath.ops"
TASK = "refactory_packages.livemath.tasks:LiveMathTask"

# Ported from skillopt/envs/livemathematicianbench/prompts/rollout_system.md (MIT)
# {skill_section} replaced by {{skill_section}} for template rendering. The file
# ends with a newline, which SkillOpt's _build_system (rollout.py:16-21) keeps.
SYSTEM_PROMPT = (
    "You are an expert mathematical reasoning agent solving multiple-choice questions.\n"
    "\n"
    "{{skill_section}}## Task Format\n"
    "You will receive one mathematics multiple-choice question and its answer choices.\n"
    "Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.\n"
    "\n"
    "## Answer Format\n"
    "Think step by step, then provide your final answer inside <answer>...</answer> tags.\n"
    "Inside the tags, output only the single choice label, such as A or C.\n"
    "\n"
    "Example:\n"
    "<answer>B</answer>\n"
)

# Arm A's exec_timeout (run-train.sh:14, run-train-seed.sh:20); SkillOpt's own
# configs/livemathematicianbench/default.yaml:17 default is 300.
SOLVE_TIMEOUT_S = 1800
# Arm A's solver output cap (configs/livemathematicianbench/default.yaml:16).
MAX_COMPLETION_TOKENS = 16384


class Solver(nn.Mode):
    """One-agent-node solver for LiveMathematicianBench items."""

    name = "solver"
    description = "solve a LiveMathematicianBench item with a single agent call"
    #: The solve node's wall limit; a subclass (or a graph edit of ``timeout_s``)
    #: changes it.
    solve_timeout_s: ClassVar[int] = SOLVE_TIMEOUT_S

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob("", "str", description="skill prompt injected into the system message"),
            # SkillOpt configs/livemathematicianbench/default.yaml:21-22
            "use_theorem": nn.Knob(False, "bool",
                                   description="include the item's theorem as a ## Theorem section"),
            "use_sketch": nn.Knob(False, "bool",
                                  description="include the item's proof sketch as a ## Proof Sketch section"),
            # SkillOpt configs/livemathematicianbench/default.yaml:16. A run
            # setting, not a search target: frozen against the optimizer, set
            # with ``--knob max_completion_tokens=N``.
            "max_completion_tokens": nn.Knob(
                MAX_COMPLETION_TOKENS, "int", lo=0, frozen=True,
                description="output-token cap on each solve-node model call; 0 = no cap"),
            # Concurrent data-node items. A resource setting, not a search
            # target: frozen against the optimizer, set with
            # ``--knob data_parallelism=N``.
            "data_parallelism": nn.Knob(
                1, "int", lo=1, hi=64, frozen=True,
                description="concurrent data-node items"),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data",
            subgraph_entry="format_prompt",
            subgraph_exit="solve",
            task_ref=TASK,
            writes=("current_item.json",),
            produces=("item_scores",),
            parallelism_knob="data_parallelism",
            knobs=("data_parallelism",),
        ))
        graph.add(nn.Node(
            "format_prompt", "op",
            call=f"{OPS}:format_prompt",
            reads=("current_item.json",),
            knobs=("skill", "use_theorem", "use_sketch"),
            produces=("user_prompt", "skill_section"),
        ))
        graph.add(nn.Node(
            "solve", "agent",
            prompt_template="{{user_prompt}}",
            system_prompt_template=SYSTEM_PROMPT,
            writes=("answer.txt",),
            tool_policy="none",
            role="solver",
            timeout_s=self.solve_timeout_s,
            knobs=("max_completion_tokens",),
        ))
        graph.add(nn.Node(
            "finish", "op",
            call=f"{OPS}:finish",
            writes=("result.json",),
            produces=("finished",),
        ))
        graph.connect("format_prompt", "solve")
        graph.connect("data", "finish")
        return graph

    def run(
        self,
        workdir: Path,
        *,
        model: Any = None,
        task: Mapping[str, Any] | None = None,
        memory: Any = None,
        **kwargs: Any,
    ) -> nn.RunResult:
        """Put the ``max_completion_tokens`` knob on the solve node's route.

        The graph has no per-node route field, so the cap travels the way any
        per-node route setting does: as the ``solve`` entry of the model's
        ``node_routes``. ``for_node("solve")`` then carries it in the worker's
        requested route, and the worker observer (``worker_observer.mjs``) sets
        it as ``maxTokens`` on the DSH ``agent/request`` config, which DSH passes
        to ``llm.prepareCall``. Other nodes' routes are untouched; ``0`` adds no
        cap.
        """
        cap = int(self.values().get("max_completion_tokens") or 0)
        if cap > 0 and model is not None and callable(getattr(model, "for_node", None)):
            routes = dict(getattr(model, "_node_routes", None) or {})
            routes["solve"] = {**routes.get("solve", {}), "max_completion_tokens": str(cap)}
            model = copy.copy(model)
            model._node_routes = routes
        return super().run(workdir, model=model, task=task, memory=memory, **kwargs)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0
