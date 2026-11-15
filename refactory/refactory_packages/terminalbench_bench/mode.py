"""TerminalBench: study → builder → gate_verify → auto_merge with reloop.

    study -> builder -> gate_verify --proceed--> auto_merge
                ^           |
                +--reloop---+

Real-world engineering tasks (compiling, scientific computing, sysadmin,
security, ML, data processing) evaluated in Harbor containers.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.terminalbench_bench.ops"


class TerminalBench(nn.Mode):
    """Solve a real-world engineering task in a terminal environment."""

    name = "terminalbench"
    description = "TerminalBench — real-world engineering tasks in containerized terminal evaluation"
    unscored_reason = "scored by Harbor's external verifier, not by the mode itself"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "attempts": nn.Knob(3, "int", lo=1, hi=5, description="how many builder iterations to allow"),
            "builder_timeout": nn.Knob(7200, "int", lo=600, hi=14400, description="seconds the builder may run"),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "study", "op", call=f"{OPS}:study",
            writes=("study-output.md",),
            produces=("study_output",),
            notes="detect installed tools and languages, read the task instruction",
        ))
        graph.add(nn.Node(
            "builder", "agent",
            prompt_template=(
                "You are solving a real-world engineering task in a terminal environment "
                "for the TerminalBench benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md for the full "
                "task description and requirements.\n\n"
                "2. **Check available tools** — review the study output for installed "
                "languages, package managers, and tools. Install any missing dependencies "
                "proactively using the available package managers (pip, apt, npm, cargo, etc.).\n\n"
                "3. **Understand the problem** — explore any provided files, configuration, "
                "or project structure. Identify what needs to be built, fixed, or configured.\n\n"
                "4. **Implement the solution** — write code, compile programs, configure "
                "systems, or perform whatever operations the task requires. Use the "
                "appropriate tools and languages for the job.\n\n"
                "5. **Verify your work** — run tests, compile binaries, check outputs, "
                "or perform whatever verification is appropriate for the task. "
                "Ensure everything works correctly before committing.\n\n"
                "6. **Commit your changes** — commit directly on the current branch "
                "with a descriptive message. Do NOT create a new branch or PR.\n\n"
                "## Rules\n\n"
                "- Install missing dependencies proactively — don't wait for errors\n"
                "- MUST verify your solution works before committing\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
                "- Do NOT run factory commands (factory eval, factory study, etc.)\n"
                "- If verification fails, investigate and fix the issue\n"
            ),
            knobs=("builder_timeout",),
            reads=("study-output.md",),
            writes=("builder-latest.md",),
            notes="solve the engineering task described in the task instruction",
        ))
        graph.add(nn.Node(
            "gate_verify", "gate", evaluator=f"{OPS}:gate_verify",
            knobs=("attempts",),
            reads=("builder-latest.md",),
            writes=("feedback.txt",),
            produces=("feedback", "attempts_used"),
            max_iterations=5,
            notes="check whether the builder committed changes and task succeeded; reloop if not",
        ))
        graph.add(nn.Node(
            "auto_merge", "op", call=f"{OPS}:auto_merge",
            reads=("builder-latest.md",),
            produces=("merge_result",),
            notes="update the main branch to include the solution",
        ))
        graph.connect("study", "builder")
        graph.connect("builder", "gate_verify")
        graph.connect("gate_verify", "builder", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
