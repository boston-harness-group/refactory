"""Salitrap: study → solver → gate_verify → auto_merge with reloop.

    study -> solver -> gate_verify --proceed--> auto_merge
                ^           |
                +--reloop---+

Commonsense reasoning under salience bias and numerical distractors,
evaluated in Harbor containers.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.salitrap_bench.ops"


class Salitrap(nn.Mode):
    """Solve a commonsense reasoning problem under salience bias."""

    name = "salitrap"
    description = "Salitrap — commonsense reasoning pipeline under salience bias"
    unscored_reason = "scored by Harbor's external verifier, not by the mode itself"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "attempts": nn.Knob(3, "int", lo=1, hi=5, description="how many solver iterations to allow"),
            "solver_timeout": nn.Knob(3600, "int", lo=600, hi=7200, description="seconds the solver may run"),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "study", "op", call=f"{OPS}:study",
            writes=("study-output.md",),
            produces=("study_output",),
            notes="discover workspace contents and read the task instruction",
        ))
        graph.add(nn.Node(
            "solver", "agent",
            prompt_template=(
                "You are solving a commonsense reasoning problem for the Salitrap benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md for the full "
                "reasoning problem.\n\n"
                "2. **Check for prerequisite violations** before attempting any calculation:\n"
                "   - Missing resources: does the scenario assume something exists that isn't stated?\n"
                "   - Environmental mismatch: is the setting incompatible with the action?\n"
                "   - Temporal/physiological violations: does the timeline or biology make sense?\n"
                "   - Rule mismatches: are any stated rules contradictory or impossible?\n\n"
                "3. **Beware of numerical distractors** — numbers in the problem may be "
                "completely irrelevant. Do NOT perform arithmetic with provided numbers "
                "without first checking whether the question actually requires calculation. "
                "The correct answer may have nothing to do with the numbers given.\n\n"
                "4. **Write your answer** to `/workspace/answer.txt` in the following format:\n"
                "   ```\n"
                "   Verdict: <your answer>\n"
                "   Trap-Type: <none | salience-bias | numerical-distractor | prerequisite-violation>\n"
                "   Reasoning: <step-by-step reasoning>\n"
                "   Key-Insight: <the critical insight that resolves the problem>\n"
                "   ```\n\n"
                "5. **Commit your changes** — commit directly on the current branch.\n\n"
                "## Rules\n\n"
                "- Think BEFORE calculating — most problems have a non-numerical answer\n"
                "- Check physical/logical prerequisites before solving\n"
                "- Write answer to /workspace/answer.txt — this is REQUIRED\n"
                "- Do NOT run factory commands (factory eval, factory study, etc.)\n"
            ),
            knobs=("solver_timeout",),
            reads=("study-output.md",),
            writes=("answer.txt",),
            notes="reason through the problem, avoiding salience traps",
        ))
        graph.add(nn.Node(
            "gate_verify", "gate", evaluator=f"{OPS}:gate_verify",
            knobs=("attempts",),
            reads=("answer.txt",),
            writes=("feedback.txt",),
            produces=("feedback", "attempts_used"),
            max_iterations=5,
            notes="check whether the solver wrote an answer and committed; reloop if not",
        ))
        graph.add(nn.Node(
            "auto_merge", "op", call=f"{OPS}:auto_merge",
            reads=("answer.txt",),
            produces=("merge_result",),
            notes="update the main branch to include the answer",
        ))
        graph.connect("study", "solver")
        graph.connect("solver", "gate_verify")
        graph.connect("gate_verify", "solver", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
