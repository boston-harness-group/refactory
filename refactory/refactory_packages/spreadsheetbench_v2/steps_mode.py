"""SpreadsheetBench 2 steps mode: V2's agent setup with its workflow lifted into four steps.

The swe mode (`swe_mode.Swe`) is one sealed Claude Code session whose prompt holds V2's
whole workflow in prose. Here that workflow is the graph::

    prepare -> has_case -> inspect -> plan -> implement -> verify -> finalize
                  |                                                    ^
                  +----------------------- skip -----------------------+

Each step is a sealed Claude Code session set up as swe's solve step (the same Task, tool
environment, seal, ``Bash`` alone, ``view_xlsx`` and LibreOffice on the PATH, a /tmp of
its own, the same retry), on the same items and grade. Its prompt is V2's instance
template with the workflow cut to its own stages and a section naming its files
(`steps_port`), each step's own open knobs ``<step>__task`` and ``<step>__system_prompt``:

- ``inspect`` reads ``input.xlsx`` and writes ``inspect_notes.md``;
- ``plan`` reads those notes (and may view the input) and writes ``plan.md``;
- ``implement`` reads both and writes ``output.xlsx`` (scripts in /tmp, as upstream);
- ``verify`` reads all of them, checks ``output.xlsx`` (recalculating with LibreOffice as
  needed), may correct it, and writes ``verify_report.md``.

A step's files reach the next only through the pass (each step's ``reads``/``writes``):
inspect and plan cannot leave an ``output.xlsx``, since they do not declare one. The
``output.xlsx`` in the pass after the last step is graded as swe's is; none fails the item
(``output-not-found``). The ``skill`` knob is one ``SKILL.md`` every step reads, and every
step's prompt names it when it is non-empty (`swe_port.skill_section`).

Run settings, as swe's and frozen: each step's turn cap is upstream's 50 (one ``max_turns``
knob bound to every step's route), each Bash command's timeout 60 s and its output cut at
10,000 characters. Four capped sessions may use up to 200 turns where swe's one uses 50:
the item's ``final.json`` records each step's turns and their total (`steps_ops.finalize`)
so cost against swe is measured, and ``tool_calls.json`` keeps every step's calls.

No loops, judges or routes: the structure beyond the workflow's split is the loop's to
find (only ``data`` and ``finalize`` are protected).
"""
from __future__ import annotations

from refactory_dsh import nn

from refactory_packages.spreadsheetbench.agent_mode import SOLVE_RETRY
from refactory_packages.spreadsheetbench.mode import OPS, SKILLOPT_CLAUDE_CODE
from refactory_packages.spreadsheetbench.tasks import INPUT

from . import steps_port, swe_mode, swe_port
from .steps_port import INSPECT_NOTES, PLAN, STEPS, VERIFY_REPORT

TASK = "refactory_packages.spreadsheetbench_v2.tasks:SpreadsheetBenchV2StepsTask"
STEPS_OPS = "refactory_packages.spreadsheetbench_v2.steps_ops"
#: The run settings every step's route takes, swe's frozen knobs.
TURNS_KNOB, TIMEOUT_KNOB, OUTPUT_KNOB = (swe_mode.TURNS_KNOB, swe_mode.TIMEOUT_KNOB,
                                         swe_mode.OUTPUT_KNOB)
#: What every step's workspace holds from prepare (copies of these pass files).
BASE_READS = swe_mode.WORKSPACE_READS
#: What each step reads besides `BASE_READS`, and what it writes after its reply: the
#: files it hands on, then its session facts.
STEP_READS = {
    "inspect": (),
    "plan": (INSPECT_NOTES,),
    "implement": (INSPECT_NOTES, PLAN),
    "verify": (INSPECT_NOTES, PLAN, swe_port.OUTPUT_PATH),
}
STEP_WRITES = {
    "inspect": (INSPECT_NOTES,),
    "plan": (PLAN,),
    "implement": (swe_port.OUTPUT_PATH,),
    "verify": (VERIFY_REPORT, swe_port.OUTPUT_PATH),
}


def task_knob(step: str) -> str:
    """The step's prompt template knob."""
    return f"{step}__task"


def system_prompt_knob(step: str) -> str:
    """The step's appended system prompt knob."""
    return f"{step}__system_prompt"


def reply_path(step: str) -> str:
    """The step's reply (its final message), its first write."""
    return f"{step}_reply.txt"


def session_path(step: str) -> str:
    """The step's session facts, collected from beside its sealed workspace
    (`workspace.is_session_log`)."""
    return f"{step}/{swe_port.SESSION_PATH}"


def step_writes(step: str) -> tuple[str, ...]:
    return (reply_path(step), *STEP_WRITES[step], session_path(step))


class Steps(nn.Mode):
    """SpreadsheetBench 2's agent setup as inspect -> plan -> implement -> verify."""

    name = "steps"
    description = ("solve a SpreadsheetBench 2 task with V2's agent setup split into its "
                   "workflow's steps: four sealed Claude Code sessions (inspect, plan, "
                   "implement, verify), each with its own prompt, hand on notes, a plan and "
                   "output.xlsx")

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        swe = swe_mode.Swe.declare()
        knobs: dict[str, nn.Knob] = {
            "skill": nn.Knob("", "str",
                             description="the skill, written to SKILL.md, which every step "
                                         "reads; empty, no prompt mentions it"),
            "data_parallelism": swe["data_parallelism"],
            TURNS_KNOB: nn.Knob(swe_port.MAX_TURNS, "int", lo=1, frozen=True,
                                description="each step's turn cap (V2's per_instance_call_"
                                            "limit, Claude Code's --max-turns)"),
            TIMEOUT_KNOB: swe[TIMEOUT_KNOB],
            OUTPUT_KNOB: swe[OUTPUT_KNOB],
        }
        for step in STEPS:
            knobs[task_knob(step)] = nn.Knob(
                steps_port.TASKS[step], "str",
                description=f"the {step} step's prompt: V2's instance template with its "
                            f"workflow cut to this step's stages")
            knobs[system_prompt_knob(step)] = nn.Knob(
                swe_port.SYSTEM_TEMPLATE, "str",
                description=f"text appended to Claude Code's system prompt for the {step} "
                            "step: V2's system template")
        return knobs

    def build(self) -> nn.Graph:
        g = nn.Graph()
        g.add(nn.Node("data", "data", subgraph_entry="prepare", subgraph_exit="finalize",
                      task_ref=TASK, writes=("current_item.json", INPUT),
                      produces=("item_scores",),
                      parallelism_knob="data_parallelism", knobs=("data_parallelism",)))
        g.add(nn.Node("prepare", "op", call=f"{STEPS_OPS}:prepare",
                      reads=("current_item.json", INPUT), knobs=("skill",),
                      writes=(*BASE_READS, "turn.json", "spreadsheet_preview.txt"),
                      produces=(*swe_mode.PROMPT_FACTS, "has_case"),
                      knob_outputs={swe_port.SKILL_PATH: ("skill",), "skill_section": ("skill",),
                                    "has_case": (), "instruction": (), "spreadsheet_path": (),
                                    "output_path": (), swe_port.INPUT_PATH: (), "turn.json": (),
                                    "spreadsheet_preview.txt": ()},
                      notes="record the item's start and lay the steps' workspace files"))
        g.add(nn.Node("has_case", "gate", evaluator=f"{OPS}:has_case", inputs=("has_case",),
                      notes="an item with no test case skips the steps"))
        for step in STEPS:
            g.add(nn.Node(step, "agent", prompt_template=steps_port.TASKS[step],
                          system_prompt_template=swe_port.SYSTEM_TEMPLATE,
                          tools=swe_port.TOOLS, workspace="scratch",
                          reads=(*BASE_READS, *STEP_READS[step]), writes=step_writes(step),
                          role="solver", retry=SOLVE_RETRY,
                          knobs=(task_knob(step), system_prompt_knob(step), TURNS_KNOB,
                                 TIMEOUT_KNOB, OUTPUT_KNOB),
                          field_bindings=(("prompt_template", task_knob(step)),
                                          ("system_prompt_template", system_prompt_knob(step)),
                                          ("route.max_turns", TURNS_KNOB),
                                          ("route.bash_timeout_ms", TIMEOUT_KNOB),
                                          ("route.bash_output_chars", OUTPUT_KNOB)),
                          notes=f"a sealed Claude Code session (Bash): the {step} step, "
                                f"writes {', '.join(STEP_WRITES[step])}"))
        g.add(nn.Node("finalize", "op", call=f"{STEPS_OPS}:finalize",
                      reads=("turn.json", swe_port.OUTPUT_PATH,
                             *(path for step in STEPS
                               for path in (reply_path(step), session_path(step)))),
                      writes=("final.json", "conversation.json", "pred.xlsx", "raw.txt",
                              "tool_calls.json"),
                      produces=("finished",),
                      notes="take the last step's output.xlsx as the item's output"))
        g.connect("prepare", "has_case")
        g.connect("has_case", STEPS[0], condition="proceed")
        g.connect("has_case", "finalize", condition="skip")
        for a, b in zip(STEPS, STEPS[1:]):
            g.connect(a, b)
        g.connect(STEPS[-1], "finalize")
        return g

    def claude_code_versions(self) -> dict[str, str]:
        """Every step runs on the Claude Code the agent mode pins."""
        return {step: SKILLOPT_CLAUDE_CODE for step in STEPS}

    def protected_nodes(self) -> tuple[str, ...]:
        """The data pass and the item's outcome."""
        return ("data", "finalize")

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0

    def guidance(self) -> str:
        return ("Four sealed steps run in order -- inspect, plan, implement, verify -- each "
                "with its own prompt (<step>__task) and system prompt (<step>__system_prompt); "
                "they hand on inspect_notes.md, plan.md and output.xlsx, and the output.xlsx "
                "left after the last step is graded. The skill is one SKILL.md every step "
                "reads. The graph can be edited too (only data and finalize are protected). "
                "Keep every change general -- it must help on items no rollout saw.")
