"""ProgramBench: builder → reviewer → gate_verify → auto_merge with reloop.

    builder -> reviewer -> gate_verify --proceed--> auto_merge
       ^                       |
       +--------reloop---------+

No study node — starts directly at builder. An extra reviewer node
adversarially validates the builder's discoveries before the gate.
"""

from __future__ import annotations

from refactory_dsh import nn

OPS = "refactory_packages.programbench_bench.ops"


class ProgramBench(nn.Mode):
    """Reverse-engineer a compiled binary and reconstruct its source."""

    name = "programbench"
    description = "ProgramBench — adversarial binary reverse-engineering pipeline"
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
            "builder", "agent",
            prompt_template=(
                "You are reverse-engineering a compiled binary for the ProgramBench benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Probe the binary** — The compiled executable is at `/workspace/executable`. "
                "A backup copy is at `/workspace/executable.bak`. Use tools like `strings`, "
                "`objdump`, `strace`, `ltrace`, `nm`, `file`, and manual execution with various "
                "inputs to understand its behavior.\n\n"
                "2. **Document discoveries** — Maintain `/workspace/discoveries.md` with all "
                "findings: input/output behavior, data structures, algorithms, edge cases, "
                "error handling. Update this file as you learn more.\n\n"
                "3. **Reconstruct the source** — Write C source code that reproduces the "
                "binary's behavior exactly. Save it alongside a `compile.sh` script "
                "that compiles it.\n\n"
                "4. **Verify your reconstruction** — Run `compile.sh`, then compare outputs "
                "of your compiled version against `executable.bak` with the same inputs. "
                "The outputs must match exactly.\n\n"
                "5. **Commit your changes** — commit all files on the current branch.\n\n"
                "## Rules\n\n"
                "- Match the binary's behavior EXACTLY — same output for same input\n"
                "- Do NOT modify `/workspace/executable.bak` — it is the ground truth\n"
                "- Keep `/workspace/discoveries.md` up to date throughout\n"
                "- Test with edge cases: empty input, large input, special characters\n"
                "- Do NOT create branches or PRs — commit on current branch\n"
            ),
            knobs=("builder_timeout",),
            writes=("discoveries.md",),
            notes="probe and reverse-engineer the compiled binary",
        ))
        graph.add(nn.Node(
            "reviewer", "agent",
            prompt_template=(
                "You are an adversarial reviewer for the ProgramBench benchmark.\n\n"
                "## Your Task\n\n"
                "1. **Read the discoveries** — Read `/workspace/discoveries.md` to understand "
                "what the builder claims about the binary's behavior.\n\n"
                "2. **Validate each discovery** — For every claim, run your own commands "
                "against both `/workspace/executable.bak` (ground truth) and the builder's "
                "compiled version to verify the claim is correct.\n\n"
                "3. **Write your review** — Save `/workspace/review.md` with your findings. "
                "For each discovery, mark it as CONFIRMED or DISPUTED with evidence.\n\n"
                "4. **Create the TODO list** — Save `/workspace/todos.md` with `## TODO` "
                "entries for anything that is incorrect or unexplored. Each TODO should be "
                "specific and actionable.\n\n"
                "## Rules\n\n"
                "- Do NOT fix the code yourself — only validate and document\n"
                "- Test with inputs the builder may not have tried\n"
                "- Be thorough — check edge cases, error paths, format details\n"
            ),
            reads=("discoveries.md",),
            writes=("review.md", "todos.md"),
            notes="adversarially validate the builder's discoveries",
        ))
        graph.add(nn.Node(
            "gate_verify", "gate", evaluator=f"{OPS}:gate_verify",
            knobs=("attempts",),
            reads=("todos.md",),
            writes=("feedback.txt", "test-results.txt"),
            produces=("feedback", "attempts_used"),
            max_iterations=5,
            notes="check TODOs and compile.sh; reloop if issues remain",
        ))
        graph.add(nn.Node(
            "auto_merge", "op", call=f"{OPS}:auto_merge",
            reads=("review.md",),
            produces=("merge_result",),
            notes="update the main branch to include the reconstruction",
        ))
        graph.connect("builder", "reviewer")
        graph.connect("reviewer", "gate_verify")
        graph.connect("gate_verify", "builder", condition="reloop")
        graph.connect("gate_verify", "auto_merge", condition="proceed")
        return graph
