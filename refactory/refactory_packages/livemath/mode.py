"""LiveMathematicianBench one-node solver mode.

The solver graph: data → [format_prompt → solve] → finish.

System prompt ported from SkillOpt (MIT) —
skillopt/envs/livemathematicianbench/prompts/rollout_system.md.
Solver model defaults from configs/_base_/default.yaml:
  model = gpt-5.5 (_base_/default.yaml:7)
  reasoning_effort = medium (_base_/default.yaml:10)
"""
from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.livemath.ops"
TASK = "refactory_packages.livemath.tasks:LiveMathTask"

# Ported from skillopt/envs/livemathematicianbench/prompts/rollout_system.md (MIT)
# {skill_section} replaced by {{skill_section}} for template rendering.
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
    "<answer>B</answer>"
)


class Solver(nn.Mode):
    """One-agent-node solver for LiveMathematicianBench items."""

    name = "solver"
    description = "solve a LiveMathematicianBench item with a single agent call"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob("", "str", description="skill prompt injected into the system message"),
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
        ))
        graph.add(nn.Node(
            "format_prompt", "op",
            call=f"{OPS}:format_prompt",
            reads=("current_item.json",),
            knobs=("skill",),
            produces=("user_prompt", "skill_section"),
        ))
        graph.add(nn.Node(
            "solve", "agent",
            prompt_template="{{user_prompt}}",
            system_prompt_template=SYSTEM_PROMPT,
            writes=("answer.txt",),
            tool_policy="none",
            role="solver",
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

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0
