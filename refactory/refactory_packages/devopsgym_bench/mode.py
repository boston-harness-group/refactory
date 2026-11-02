"""DevOpsGym: study → solver → gate_verify → auto_merge with reloop.

    study -> solver -> gate_verify --proceed--> auto_merge
               ^           |
               +--reloop---+

Build and configuration tasks (Maven, Gradle, Go, Make, Docker, CI/CD)
evaluated in Harbor containers.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.devopsgym_bench.ops"


class DevOpsGym(nn.Mode):
    """Solve a build or configuration task in a containerized environment."""

    name = "devopsgym"
    description = "DevOpsGym — build/configuration task pipeline for containerized evaluation"
    unscored_reason = "scored by Harbor's external verifier, not by the mode itself"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "attempts": nn.Knob(3, "int", lo=1, hi=5, description="how many solver iterations to allow"),
            "solver_timeout": nn.Knob(7200, "int", lo=600, hi=14400, description="seconds the solver may run"),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "study", "op", call=f"{OPS}:study",
            writes=("study-output.md",),
            produces=("study_output",),
            notes="discover build system configuration and read the task instruction",
        ))
        graph.add(nn.Node(
            "solver", "agent",
            prompt_template=(
                "You are solving a build or configuration task for the DevOpsGym benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Read the task instruction** — Read /tmp/task-instruction.md for the full "
                "task description and requirements.\n\n"
                "2. **Understand the build system** — explore the repository structure. "
                "Identify the build system in use (Maven, Gradle, Go, Make, Docker, "
                "CI/CD pipelines, etc.). Read build files, configuration, and any "
                "documentation about the build process.\n\n"
                "3. **Solve the task** — make the changes needed to fix the build or "
                "configuration issue. PRESERVE the existing build system — do NOT switch "
                "to a different tool (e.g. do not replace Maven with Gradle, or Make with "
                "a shell script). Work within the project's chosen toolchain.\n\n"
                "4. **Verify the build** — this is CRITICAL. Run the project's build "
                "command to verify your fix works:\n"
                "   - Maven: `mvn verify`\n"
                "   - Gradle: `gradle build`\n"
                "   - Go: `go build ./...`\n"
                "   - Make: `make`\n"
                "   - Docker: `docker build .`\n"
                "   Note: there may be hidden verification steps beyond the basic build "
                "command. Ensure the full build pipeline succeeds.\n\n"
                "5. **Commit your changes** — commit directly on the current branch "
                "with a descriptive message. Do NOT create a new branch. Do NOT create a PR.\n\n"
                "## Rules\n\n"
                "- PRESERVE the existing build system — do NOT switch tools\n"
                "- MUST run the build before committing — never commit unverified changes\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
                "- Do NOT run factory commands (factory eval, factory study, etc.)\n"
                "- If the build fails after your fix, investigate and fix the issue\n"
            ),
            knobs=("solver_timeout",),
            reads=("study-output.md",),
            writes=("builder-latest.md",),
            notes="solve the build/configuration task described in the task instruction",
        ))
        graph.add(nn.Node(
            "gate_verify", "gate", evaluator=f"{OPS}:gate_verify",
            knobs=("attempts",),
            reads=("builder-latest.md",),
            writes=("feedback.txt",),
            produces=("feedback", "attempts_used"),
            max_iterations=5,
            notes="attempt the project build; reloop if it fails",
        ))
        graph.add(nn.Node(
            "auto_merge", "op", call=f"{OPS}:auto_merge",
            reads=("builder-latest.md",),
            produces=("merge_result",),
            notes="update the main branch to include the fix",
        ))
        graph.connect("study", "solver")
        graph.connect("solver", "gate_verify")
        graph.connect("gate_verify", "solver", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
