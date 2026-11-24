"""SpreadsheetBench 2 swe mode: V2's own agent setup (its SWE-agent config) as a sealed Claude Code step.

Per item, one sealed Claude Code session (`refactory_dsh.seal`, `workspace.is_sealed`) with
``Bash`` in a ``scratch`` workspace holding ``input.xlsx`` (and ``SKILL.md``),
on the same route plumbing as the agent mode (`spreadsheetbench.agent_mode`): each Bash
command runs under the seal (the workspace only, no network), with the pinned interpreter
as ``python3``, upstream's ``view_xlsx`` and LibreOffice on its PATH and a /tmp of its own
(`tasks.SpreadsheetBenchV2SweTask`). Its prompt is V2's instance template, its appended
system prompt V2's system template (`swe_port`, which lists every deviation from upstream
with its spreadsheet.yaml line). The agent edits the workbook itself and writes
``output.xlsx``, which is graded by V2's grade (LibreOffice's recalculation, then
``evaluation.py``) as the agent mode's output is; no ``output.xlsx`` fails the item.

Run settings: the route's, as in the agent mode (the ``claude_code`` worker, permission mode
``skip``, Claude Code 2.1.299); the turn cap is upstream's call cap, 50, the frozen knob
``max_turns`` bound to the solve step's route (`nn.Node.bound_route`, ``--max-turns``) on
both runtimes. A session stopped there ends with the adapter's note as its answer, the
``output.xlsx`` it left is graded, and the item's record says so (`swe_ops.finalize`,
`tasks.SpreadsheetBenchV2SweTask.verify`). Each Bash command's timeout is upstream's 60 s and
a command's output past upstream's 10,000 characters is cut (the frozen knobs
``bash_timeout_ms`` and ``bash_output_chars``, bound alike). A crashed or empty session gets
one more, in a fresh workspace, as in the agent mode. No gold feedback reaches the agent (``Task.train_feedback``
is not offered).

The one open knob is ``skill``, written to ``SKILL.md``; empty by default (upstream has
none), and then the prompt does not mention it (`swe_port.skill_section`).
"""
from __future__ import annotations

from refactory_dsh import nn

from refactory_packages.spreadsheetbench.agent_mode import SOLVE_RETRY
from refactory_packages.spreadsheetbench.mode import (OPS, SKILLOPT_CLAUDE_CODE, author_note,
                                                       upgrade_frozen_descriptions)
from refactory_packages.spreadsheetbench.tasks import INPUT

from . import swe_port

TASK = "refactory_packages.spreadsheetbench_v2.tasks:SpreadsheetBenchV2SweTask"
SWE_OPS = "refactory_packages.spreadsheetbench_v2.swe_ops"
#: The solve step's prompt (upstream's instance template), frozen: the knob it is bound to.
PROMPT_KNOB = "solve__prompt_template"
#: The text appended to Claude Code's system prompt (upstream's system template), frozen.
SYSTEM_PROMPT_KNOB = "solve__system_prompt"
#: The solve step's turn cap, upstream's call cap (frozen).
TURNS_KNOB = "max_turns"
#: Each Bash command's timeout and the most of its output the model sees, upstream's (frozen).
TIMEOUT_KNOB = "bash_timeout_ms"
OUTPUT_KNOB = "bash_output_chars"
#: Concurrent items, as the agent mode's (each grade runs LibreOffice on a large workbook).
DATA_PARALLELISM = 8
#: What the solve step's workspace holds (copies of these pass files, by their names).
WORKSPACE_READS = (swe_port.INPUT_PATH, swe_port.SKILL_PATH)
#: The prompt's slots, filled from prepare's facts.
PROMPT_FACTS = ("instruction", "spreadsheet_path", "output_path", "skill_section")


class Swe(nn.Mode):
    """SpreadsheetBench 2's own agent setup over its tasks."""

    name = "swe"
    description = ("solve a SpreadsheetBench 2 task with V2's own agent setup (its SWE-agent "
                   "prompt and tools): one sealed Claude Code session edits the workbook and "
                   "writes output.xlsx")

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob("", "str",
                             description="the skill, written to the agent's SKILL.md; empty, "
                                         "the prompt does not mention it"),
            "data_parallelism": nn.Knob(DATA_PARALLELISM, "int", lo=1, hi=64, frozen=True,
                                        description="concurrent items"),
            TURNS_KNOB: nn.Knob(swe_port.MAX_TURNS, "int", lo=1, frozen=True,
                                description="the session's turn cap (V2's per_instance_call_limit"
                                            ", Claude Code's --max-turns)"),
            TIMEOUT_KNOB: nn.Knob(swe_port.BASH_TIMEOUT_MS, "int", lo=1, frozen=True,
                                  description="each Bash command's timeout in ms (V2's "
                                              "execution_timeout)"),
            OUTPUT_KNOB: nn.Knob(swe_port.BASH_OUTPUT_CHARS, "int", lo=1, frozen=True,
                                 description="the most of a command's output the model sees "
                                             "(V2's max_observation_length)"),
            PROMPT_KNOB: nn.Knob(swe_port.INSTANCE_TEMPLATE, "str", frozen=True,
                                 description="the solve call's prompt, V2's instance template "
                                             "(frozen)"),
            SYSTEM_PROMPT_KNOB: nn.Knob(swe_port.SYSTEM_TEMPLATE, "str", frozen=True,
                                        description="text appended to Claude Code's system "
                                                    "prompt: V2's system template (frozen)"),
        }

    def build(self) -> nn.Graph:
        g = nn.Graph()
        g.add(nn.Node("data", "data", subgraph_entry="prepare", subgraph_exit="finalize",
                      task_ref=TASK, writes=("current_item.json", INPUT),
                      produces=("item_scores",),
                      parallelism_knob="data_parallelism", knobs=("data_parallelism",)))
        g.add(nn.Node("prepare", "op", call=f"{SWE_OPS}:prepare",
                      reads=("current_item.json", INPUT), knobs=("skill",),
                      writes=(*WORKSPACE_READS, "turn.json", "target_system_prompt.txt",
                              "target_user_prompt.txt", "spreadsheet_preview.txt"),
                      produces=(*PROMPT_FACTS, "has_case"),
                      knob_outputs={swe_port.SKILL_PATH: ("skill",), "skill_section": ("skill",),
                                    "target_user_prompt.txt": ("skill",), "has_case": (),
                                    "instruction": (), "spreadsheet_path": (), "output_path": (),
                                    swe_port.INPUT_PATH: (), "turn.json": (),
                                    "target_system_prompt.txt": (), "spreadsheet_preview.txt": ()},
                      notes="record the item's start and lay the session's workspace files"))
        g.add(nn.Node("has_case", "gate", evaluator=f"{OPS}:has_case", inputs=("has_case",),
                      notes="an item with no test case skips the call"))
        g.add(nn.Node("solve", "agent", prompt_template=swe_port.INSTANCE_TEMPLATE,
                      system_prompt_template=swe_port.SYSTEM_TEMPLATE,
                      tools=swe_port.TOOLS, workspace="scratch", reads=WORKSPACE_READS,
                      writes=("reply.txt", swe_port.OUTPUT_PATH, swe_port.SESSION_PATH),
                      role="solver", retry=SOLVE_RETRY,
                      knobs=(PROMPT_KNOB, SYSTEM_PROMPT_KNOB, TURNS_KNOB, TIMEOUT_KNOB,
                             OUTPUT_KNOB),
                      field_bindings=(("prompt_template", PROMPT_KNOB),
                                      ("system_prompt_template", SYSTEM_PROMPT_KNOB),
                                      ("route.max_turns", TURNS_KNOB),
                                      ("route.bash_timeout_ms", TIMEOUT_KNOB),
                                      ("route.bash_output_chars", OUTPUT_KNOB)),
                      notes="one sealed Claude Code session (Bash) writes output.xlsx"))
        g.add(nn.Node("finalize", "op", call=f"{SWE_OPS}:finalize",
                      reads=("turn.json", "reply.txt", swe_port.OUTPUT_PATH,
                             swe_port.SESSION_PATH),
                      writes=("final.json", "conversation.json", "pred.xlsx", "raw.txt",
                              "tool_calls.json"),
                      produces=("finished",),
                      notes="take the session's output.xlsx as the item's output"))
        g.connect("prepare", "has_case")
        g.connect("has_case", "solve", condition="proceed")
        g.connect("has_case", "finalize", condition="skip")
        g.connect("solve", "finalize")
        return g

    def claude_code_versions(self) -> dict[str, str]:
        """The solve step runs on the Claude Code the agent mode pins."""
        return {"solve": SKILLOPT_CLAUDE_CODE}

    @classmethod
    def upgrade_state(cls, state):
        """A saved state's frozen knob descriptions as this version words them
        (`spreadsheetbench.mode.upgrade_frozen_descriptions`)."""
        return upgrade_frozen_descriptions(state)

    def protected_nodes(self) -> tuple[str, ...]:
        """The data pass and the item's outcome."""
        return ("data", "finalize")

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0

    def guidance(self) -> str:
        return author_note(self, "The `skill` knob is written to the agent's SKILL.md, which "
                                 "the prompt tells it to read before it edits the workbook (an "
                                 "empty skill is not mentioned).")
