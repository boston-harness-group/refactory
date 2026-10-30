"""TomSWE: study → builder → gate_verify → auto_merge with reloop.

    study -> builder -> gate_verify --proceed--> auto_merge
                ^           |
                +--reloop---+

Preference-aware tasks with deliberately vague instructions and user
profiles.  The builder must infer intent and apply the user's coding
style preferences.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.tomswe_bench.ops"


class TomSWE(nn.Mode):
    """Solve a preference-aware task matching a user's coding style."""

    name = "tomswe"
    description = "TomSWE — preference-aware task pipeline with user profiles"
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
                "You are completing a coding task for a specific user whose preferences "
                "are described in the task instruction.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md. It contains "
                "BOTH the task description AND a user profile section.\n\n"
                "2. **Extract the user profile** — The user profile describes the user's coding "
                "preferences: naming conventions (snake_case vs camelCase, etc.), test style "
                "(pytest vs unittest, TDD vs after-the-fact), git workflow (commit message "
                "format, branch naming), and documentation preferences (docstrings, comments, "
                "README updates).\n\n"
                "3. **Infer intent from vague instructions** — The task description is "
                "deliberately vague. Use the codebase context and user profile to infer "
                "what the user actually wants. Look at existing code patterns, naming "
                "conventions already in use, and the user's stated preferences to guide "
                "your interpretation.\n\n"
                "4. **Implement the task** — Apply the user's coding style preferences "
                "throughout your implementation:\n"
                "   - Use the naming conventions they prefer\n"
                "   - Write tests in their preferred style (if tests are expected)\n"
                "   - Add documentation matching their preferences\n"
                "   - Follow their preferred code organization patterns\n\n"
                "5. **Run tests** — Run the project's test suite to verify your changes "
                "work and existing tests still pass.\n\n"
                "6. **Commit your changes** — Commit following the user's git convention "
                "(commit message style, format). Commit directly on the current branch. "
                "Do NOT create a new branch. Do NOT create a PR.\n\n"
                "## Rules\n\n"
                "- Match the user's coding style — this is the PRIMARY evaluation criterion\n"
                "- Infer intent — do not ask for clarification, make reasonable choices\n"
                "- MUST run tests before committing — never commit untested code\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
                "- Do NOT run factory commands (factory eval, factory study, etc.)\n"
            ),
            knobs=("builder_timeout",),
            reads=("study-output.md",),
            writes=("builder-latest.md",),
            notes="implement the task matching the user's coding style preferences",
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
            notes="update the main branch to include the changes",
        ))
        graph.connect("study", "builder")
        graph.connect("builder", "gate_verify")
        graph.connect("gate_verify", "builder", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
