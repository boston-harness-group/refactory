"""FeatureBench: study → builder → gate_verify → auto_merge with reloop.

    study -> builder -> gate_verify --proceed--> auto_merge
                ^           |
                +--reloop---+

Feature implementation benchmark with explicit interface specs,
evaluated in Harbor containers.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.featurebench_bench.ops"


class FeatureBench(nn.Mode):
    """Implement features matching explicit interface specifications."""

    name = "featurebench"
    description = "FeatureBench — feature implementation pipeline with interface specs"
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
            notes="discover repository structure, placeholders, and read the task instruction",
        ))
        graph.add(nn.Node(
            "builder", "agent",
            prompt_template=(
                "You are implementing a feature in a Python project for the FeatureBench benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md for the full "
                "feature specification, including required function signatures, class interfaces, "
                "and import paths.\n\n"
                "2. **Understand the codebase** — explore the repository structure. "
                "Read relevant source files, test files, and configuration. "
                "Pay special attention to existing interfaces, abstract base classes, "
                "and placeholder implementations (NotImplementedError).\n\n"
                "3. **Implement the feature** — fill in placeholder implementations and "
                "add new code as required by the spec. You MUST:\n"
                "   - Match required function signatures EXACTLY (name, parameters, return type)\n"
                "   - Use the exact import paths specified in the task\n"
                "   - Handle cross-file dependencies carefully — if module A imports from "
                "module B, ensure B's exports exist before A tries to use them\n"
                "   - Watch for NameError, ImportError, and TypeError — these indicate "
                "signature or import mismatches\n\n"
                "4. **Run the project's own tests** — this is CRITICAL. Run the test "
                "suite to verify your implementation works AND existing tests still pass. "
                "Use pytest, tox, or whatever test runner the project uses. "
                "If specific test files are mentioned in the task, run those first.\n\n"
                "5. **Commit your changes** — commit directly on the current branch "
                "with a descriptive message. Do NOT create a new branch. Do NOT create a PR.\n\n"
                "## Rules\n\n"
                "- Match interface specs EXACTLY — wrong signatures cause test failures\n"
                "- Handle cross-file imports carefully — missing exports cause ImportError\n"
                "- MUST run tests before committing — never commit untested code\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
                "- Do NOT run factory commands (factory eval, factory study, etc.)\n"
                "- If tests fail after your implementation, investigate and fix the issue\n"
            ),
            knobs=("builder_timeout",),
            reads=("study-output.md",),
            writes=("builder-latest.md",),
            notes="implement the feature described in the task instruction",
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
            notes="update the main branch to include the implementation",
        ))
        graph.connect("study", "builder")
        graph.connect("builder", "gate_verify")
        graph.connect("gate_verify", "builder", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
