"""SWE-bench: study → builder → gate_verify → auto_merge with reloop on test failure.

    study -> builder -> gate_verify --proceed--> auto_merge
                ^           |
                +--reloop---+

Designed for Harbor containers where the task instruction is at
/tmp/task-instruction.md and Harbor's pytest verifier is the final
authority on pass/fail.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.swebench_bench.ops"


class SWEBench(nn.Mode):
    """Solve a SWE-bench issue in a containerized environment."""

    name = "swebench"
    description = "SWE-bench benchmark — minimal bug-fix pipeline for containerized evaluation"
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
            notes="discover repository structure and read the task instruction",
        ))
        graph.add(nn.Node(
            "builder", "agent",
            prompt_template=(
                "You are fixing a bug in an open-source project for the SWE-bench benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md for the full "
                "bug description and task requirements.\n\n"
                "2. **Understand the codebase** — explore the repository structure. "
                "Read relevant source files, test files, and configuration. "
                "Identify the root cause of the bug described in the task.\n\n"
                "3. **Implement the fix** — make the MINIMAL change that resolves the "
                "issue. Do NOT refactor, modernize, or add unrelated improvements. "
                "Fix ONLY the described bug.\n\n"
                "4. **Run the project's own tests** — this is CRITICAL. Run the test "
                "suite to verify your fix works AND existing tests still pass. "
                "Use pytest, tox, or whatever test runner the project uses. "
                "If specific test files are mentioned in the task, run those first.\n\n"
                "5. **Commit your changes** — commit directly on the current branch "
                "with a descriptive message referencing the issue. Do NOT create a "
                "new branch. Do NOT create a PR.\n\n"
                "## Rules\n\n"
                "- MINIMAL fix only — smallest diff that resolves the issue\n"
                "- MUST run tests before committing — never commit untested code\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
                "- Do NOT run factory commands (factory eval, factory study, etc.)\n"
                "- Do NOT modify test files unless the bug is IN the test infrastructure\n"
                "- If tests fail after your fix, investigate and fix the issue\n"
            ),
            knobs=("builder_timeout",),
            reads=("study-output.md",),
            writes=("builder-latest.md",),
            notes="fix the bug described in the task instruction",
        ))
        graph.add(nn.Node(
            "gate_verify", "gate", evaluator=f"{OPS}:gate_verify",
            knobs=("attempts",),
            reads=("builder-latest.md",),
            writes=("feedback.txt",),
            produces=("feedback", "attempts_used"),
            max_iterations=5,
            notes="check whether the builder committed changes and tests pass; reloop if not",
        ))
        graph.add(nn.Node(
            "auto_merge", "op", call=f"{OPS}:auto_merge",
            reads=("builder-latest.md",),
            produces=("merge_result",),
            notes="update the main branch to include the fix",
        ))
        graph.connect("study", "builder")
        graph.connect("builder", "gate_verify")
        graph.connect("gate_verify", "builder", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
