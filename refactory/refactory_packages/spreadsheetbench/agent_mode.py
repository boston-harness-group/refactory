"""SpreadsheetBench agent mode: SkillOpt's Claude Code agent harness as a graph.

SkillOpt's ``claude_code_exec`` target backend with SpreadsheetBench mode ``single``
(skillopt/envs/spreadsheetbench/adapter.py:70-76 refuses another mode with it): the paper's
"Claude Code" harness (arXiv 2605.23904, §4, Table 1). Per item, one Claude Code session with
``Read`` and ``Bash`` works in a folder holding ``input.xlsx``, the skill as
``.agents/skills/skillopt-target/SKILL.md``, ``task.md`` and ``run_solution.py``; it may
inspect the input and run ``python run_solution.py``, and writes ``solution.py``. That code
(else the reply's first fenced block) is run on the input and graded as the solver's is
(`agent_ops`, `agent_port`). No gold feedback reaches the agent: SkillOpt's single path
takes none.

The solve step is a sealed Claude Code step (`refactory_dsh.seal`, `workspace.is_sealed`):
``tools=("Read", "Bash")`` in a ``scratch`` workspace, its reads copied in. The adapter
offers and allows those two tools only, as SkillOpt's ``--tools Read,Bash --allowedTools
Read,Bash``; each Bash command runs under the seal (the workspace only, no network, the
pinned interpreter as ``python``, `tasks.SpreadsheetBenchTask.tool_environment`), and a
Read outside the workspace is refused. Its prompt is SkillOpt's ``_exec_prompt`` of the
workspace prompt, byte for byte, with Claude Code's own system prompt and nothing appended,
as SkillOpt's CLI sends it; no execution contract is added (`agent_delivery`).

Run settings, as SkillOpt's harness passes them, are the route's, as in the solver: the
``claude_code`` worker, ``claude-opus-4-6`` at effort ``medium``, permission mode ``skip``
(Claude Code's ``bypassPermissions``, SkillOpt's) -- a sealed Claude Code step is refused
under any other. The thinking budget (``--max-thinking-tokens 16384``) is the frozen knob
``max_thinking_tokens``, bound to the solve step's route (`nn.Node.bound_route`), so it
reaches the route on both runtimes. Claude Code is pinned to 2.1.299, the lab's install
that SkillOpt's runs used (SkillOpt pins none; `Agent.claude_code_versions`).

**Deviations from SkillOpt as shipped** (docs/spreadsheetbench.md lists each with its reason):

1. *Sealed workspace*: SkillOpt's agent runs unconfined in a folder of its run (its Bash
   and Read reach the dataset, the golden files, other items' outputs and the network);
   here it sees its workspace only.
2. *Interpreter*: ``python`` in the workspace is the pinned interpreter model code runs
   under; SkillOpt's is whatever the agent's shell finds, while it grades with its own.
3. *No item budget, no per-call wall* (SkillOpt: 600 s per item for the call and the
   grading together), as in the solver.
4. *Settings*: none loaded (no user or project settings, CLAUDE.md or MCP servers);
   SkillOpt's CLI loads the operator's.
5. *Retry*: a crashed or empty session gets one more, in a fresh workspace (SkillOpt:
   one more ``claude -p`` in the same folder, its prompt told the last answer was empty);
   an empty last turn is first asked once, in the session, for its answer (the adapter).
6. *Grader*: ``recalc`` by default, as in the solver.
"""
from __future__ import annotations

from typing import Any

from refactory_dsh import nn

from . import agent_port
from . import skillopt_port as port
from .mode import OPS, SKILLOPT_CLAUDE_CODE

#: The same items and grade, with the input's preview as each item's context for the
#: optimizer (`tasks.SpreadsheetBenchAgentTask`).
TASK = "refactory_packages.spreadsheetbench.tasks:SpreadsheetBenchAgentTask"
from .tasks import INPUT

AGENT_OPS = "refactory_packages.spreadsheetbench.agent_ops"
#: The solve step's prompt, frozen as SkillOpt's is: the knob its template is bound to.
PROMPT_KNOB = "solve__prompt_template"
#: The solve step's system prompt: Claude Code's own, nothing appended, as SkillOpt's CLI
#: sends it (no ``--append-system-prompt``); frozen, so no edit appends one.
SYSTEM_PROMPT_KNOB = "solve__system_prompt"
#: The solve step's thinking budget, SkillOpt's ``--max-thinking-tokens`` (frozen).
THINKING_KNOB = "max_thinking_tokens"
#: SkillOpt's one retry on an empty response (``EXEC_EMPTY_RESPONSE_RETRIES``).
SOLVE_RETRY = {"on": ["crash"], "times": agent_port.EMPTY_RESPONSE_RETRIES}
#: What the solve step's workspace holds (copies of these pass files, by their names).
WORKSPACE_READS = (agent_port.INPUT_PATH, agent_port.SKILL_PATH, agent_port.TASK_PATH,
                   agent_port.DRIVER_PATH)


class Agent(nn.Mode):
    """SkillOpt's SpreadsheetBench agent harness over the Task's items."""

    name = "agent"
    description = ("solve a SpreadsheetBench item as SkillOpt's Claude Code agent harness: one "
                   "sealed Claude Code session with Read and Bash writes solution.py")
    #: The Task whose items the data pass runs, and the module of its prepare and finalize
    #: ops: another dataset reuses this harness by naming its own (`spreadsheetbench_v2`).
    task_ref = TASK
    ops_module = AGENT_OPS

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob(port.INITIAL_SKILL, "str",
                             description="the skill, written to the agent's SKILL.md"),
            "data_parallelism": nn.Knob(24, "int", lo=1, hi=64, frozen=True,
                                        description="concurrent items (SkillOpt's workers)"),
            THINKING_KNOB: nn.Knob(agent_port.MAX_THINKING_TOKENS, "int", lo=1, frozen=True,
                                   description="the solve call's thinking budget (SkillOpt's "
                                               "--max-thinking-tokens)"),
            PROMPT_KNOB: nn.Knob(agent_port.SOLVE_PROMPT, "str", frozen=True,
                                 description="the solve call's prompt (frozen; use the skill "
                                             "knob)"),
            SYSTEM_PROMPT_KNOB: nn.Knob("", "str", frozen=True,
                                        description="text appended to Claude Code's system prompt "
                                                    "for the solve call: none, as SkillOpt's CLI "
                                                    "(frozen; use the skill knob)"),
        }

    def build(self) -> nn.Graph:
        g = nn.Graph()
        g.add(nn.Node("data", "data", subgraph_entry="prepare", subgraph_exit="finalize",
                      task_ref=self.task_ref, writes=("current_item.json", INPUT),
                      produces=("item_scores",),
                      parallelism_knob="data_parallelism", knobs=("data_parallelism",)))
        g.add(nn.Node("prepare", "op", call=f"{self.ops_module}:prepare",
                      reads=("current_item.json", INPUT), knobs=("skill",),
                      writes=(*WORKSPACE_READS, "turn.json", "target_system_prompt.txt",
                              "target_user_prompt.txt", "spreadsheet_preview.txt"),
                      produces=("has_case",),
                      knob_outputs={agent_port.SKILL_PATH: ("skill",), "has_case": (),
                                    "target_system_prompt.txt": ("skill",),
                                    agent_port.INPUT_PATH: (), agent_port.TASK_PATH: (),
                                    agent_port.DRIVER_PATH: (), "turn.json": (),
                                    "target_user_prompt.txt": (), "spreadsheet_preview.txt": ()},
                      notes="record the item's start and write SkillOpt's workspace files"))
        g.add(nn.Node("has_case", "gate", evaluator=f"{OPS}:has_case", inputs=("has_case",),
                      notes="an item with no test case skips the call"))
        g.add(nn.Node("solve", "agent", prompt_template=agent_port.SOLVE_PROMPT,
                      tools=agent_port.TOOLS, workspace="scratch", reads=WORKSPACE_READS,
                      writes=("reply.txt", agent_port.SOLUTION_PATH), role="solver",
                      retry=SOLVE_RETRY, knobs=(PROMPT_KNOB, SYSTEM_PROMPT_KNOB, THINKING_KNOB),
                      field_bindings=(("prompt_template", PROMPT_KNOB),
                                      ("system_prompt_template", SYSTEM_PROMPT_KNOB),
                                      ("route.max_thinking_tokens", THINKING_KNOB)),
                      notes="one sealed Claude Code session (Read, Bash) writes solution.py"))
        g.add(nn.Node("finalize", "op", call=f"{self.ops_module}:finalize",
                      reads=("current_item.json", "turn.json", "reply.txt",
                             agent_port.SOLUTION_PATH, INPUT),
                      writes=("final.json", "conversation.json", "pred.xlsx", "code.py",
                              "raw.txt"),
                      produces=("finished",),
                      notes="run the solution and record the item's outcome before grading"))
        g.connect("prepare", "has_case")
        g.connect("has_case", "solve", condition="proceed")
        g.connect("has_case", "finalize", condition="skip")
        g.connect("solve", "finalize")
        return g

    def claude_code_versions(self) -> dict[str, str]:
        """The solve step runs on SkillOpt's Claude Code, as the solver's does."""
        return {"solve": SKILLOPT_CLAUDE_CODE}

    def protected_nodes(self) -> tuple[str, ...]:
        """The data pass and the item's outcome."""
        return ("data", "finalize")

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0

    def guidance(self) -> str:
        return ("The skill is written to the agent's SKILL.md, which it reads before writing "
                "solution.py for every item; the graph can be edited too -- the steps before "
                "and around the solve step (only data and finalize are protected). Keep every "
                "change general -- it must help on items no rollout saw.")
