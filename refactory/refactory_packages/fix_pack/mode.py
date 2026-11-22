"""Make a failing check pass: one coder, the check, and another attempt until it passes.

    prepare -> emit -> materialise -> check --proceed--> report
                ^                       |
                +--------reloop---------+

Everything that places the run and lands its change is `repo_change`'s: the knobs that
say where it works, the facts its prepare op returns, and the materialise node. What is
the mode's own is its prompt, its check gate and its report. It opens no pull request: a
pull request is opened only behind an approval gate that names it (#350), so the fix stays
on the run's branch, which the report names.
"""

from __future__ import annotations

from refactory_dsh import nn, repo_change

OPS = "refactory_packages.fix_pack.ops"


class MakePass(nn.Mode):
    """Change a repository until its check passes."""

    name = "make_pass"
    description = "change a repository until a failing check passes, on a branch of its own"
    unscored_reason = "it is judged by whether the check it was given passes, which its own gate requires"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "name": nn.Knob("", "str", frozen=True,
                            description="what the run is called; a new project is named after it"),
            "brief": nn.Knob("", "str", frozen=True, description="what the fix should do, if the check does not say"),
            "check": nn.Knob("", "str", frozen=True, description="the command to make pass"),
            "attempts": nn.Knob(3, "int", lo=1, hi=5, description="how many fixes may be tried"),
            "check_timeout_s": nn.Knob(300, "int", lo=10, hi=3600, description="inactivity limit for the check (seconds of no output); the shared backstop is the absolute wall cap"),
            **repo_change.knobs(),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "prepare", "op", call=f"{OPS}:prepare",
            knobs=("name", "brief", "check", *repo_change.knob_names()),
            writes=("feedback.txt", "brief.json"),
            produces=("name", "brief", "check", "feedback", *repo_change.produces()),
            notes="read the check, and give the run its worktree",
        ))
        graph.add(nn.Node(
            # Named `emit`, so the target guard holds the agent to emit.txt (`repo_change.OUTPUT_ONLY`).
            "emit", "agent",
            prompt_template=(
                "Make this check pass: {{check}}\n"
                "It runs from {{check_root}}.\n"
                "What is wanted: {{brief}}\n\n"
                "## What the last check said\n{{feedback}}\n\n"
                "{{emit_scope}}\n\n"
                "Write those blocks into emit.txt, with no prose outside them. Do not create or "
                "change the files themselves: the materialise step applies emit.txt, and tells the "
                "next attempt if it could not. Then return a short completion message.\n"
            ),
            reads=("feedback.txt",),
            writes=("emit.txt",),
            produces=("emit",),
            notes="write the fix as blocks in emit.txt",
        ))
        graph.add(repo_change.materialise_node())
        graph.add(nn.Node(
            "check", "gate", evaluator=f"{OPS}:check",
            knobs=("attempts", "check_timeout_s"),
            # What `repo_change.run_check` may move: an install's lockfile amends the attempt.
            writes=("feedback.txt", *repo_change.CHECK_WRITES),
            produces=("feedback", "attempts_used", *repo_change.CHECK_PRODUCES),
            max_iterations=5,
            notes="run the check where the run works; reloop with what it said until it passes",
        ))
        graph.add(nn.Node(
            "report", "op", call=f"{OPS}:report",
            writes=("result.json",),
            produces=("result",),
            notes="say which branch holds the fix",
        ))
        graph.connect("prepare", "emit")
        graph.connect("emit", "materialise")
        graph.connect("materialise", "check")
        graph.connect("check", "emit", condition="reloop")
        graph.connect("check", "report", condition="proceed")
        # Independent verification step: any mode that uses repo_change gets the verifier.
        from refactory_dsh.verification import wire_verification
        wire_verification(graph, check_node="check", proceed_target="report")
        return graph
