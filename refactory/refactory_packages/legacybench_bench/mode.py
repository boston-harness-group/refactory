"""LegacyBench: study → builder → gate_verify → auto_merge with reloop.

    study -> builder -> gate_verify --proceed--> auto_merge
                ^           |
                +--reloop---+

Legacy code bugs (COBOL, Fortran, C, Java 7, Assembly) evaluated in
Harbor containers. The gate runs `make` and `make test` directly rather
than grepping builder output.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.legacybench_bench.ops"


class LegacyBench(nn.Mode):
    """Fix a bug in legacy code (COBOL, Fortran, C, Java 7, Assembly)."""

    name = "legacybench"
    description = "LegacyBench — legacy code bug-fix pipeline for multi-language containerized evaluation"
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
            notes="discover multi-language repository structure and check build state",
        ))
        graph.add(nn.Node(
            "builder", "agent",
            prompt_template=(
                "You are fixing a bug in a legacy codebase. The project may use "
                "COBOL, Fortran, C, Java 7, Assembly, or other older languages.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md for the full "
                "bug description and task requirements.\n\n"
                "2. **Understand the codebase** — explore the repository. Read source files, "
                "test files, and build configuration (Makefile, build scripts). "
                "Identify the root cause of the bug.\n\n"
                "3. **Implement the fix** — make the MINIMAL change that resolves the issue.\n\n"
                "## Critical Rules for Legacy Code\n\n"
                "- **Preserve the original language standard** — do NOT modernize the code. "
                "If it is COBOL-85, keep COBOL-85 style. If it is Fortran 77, keep F77 style. "
                "Do NOT introduce modern language features.\n"
                "- **Do NOT hardcode outputs** — hidden test inputs exist beyond the visible "
                "test cases. Your fix must be general, not tuned to specific test values.\n"
                "- **Output format precision matters** — field widths, decimal places, spacing, "
                "and alignment must match exactly. A correct value in the wrong format is a failure.\n"
                "- **Run `make test` before committing** — the build system is the authority. "
                "If `make test` fails, your fix is wrong.\n"
                "- MINIMAL fix only — smallest diff that resolves the issue\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
                "- Do NOT run factory commands\n"
                "- Do NOT modify test files unless the bug is IN the test infrastructure\n"
            ),
            knobs=("builder_timeout",),
            reads=("study-output.md",),
            writes=("builder-latest.md",),
            notes="fix the legacy code bug described in the task instruction",
        ))
        graph.add(nn.Node(
            "gate_verify", "gate", evaluator=f"{OPS}:gate_verify",
            knobs=("attempts",),
            reads=("builder-latest.md",),
            writes=("feedback.txt",),
            produces=("feedback", "attempts_used"),
            max_iterations=5,
            notes="run make and make test; reloop on compilation or test failure",
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
