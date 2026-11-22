"""SpreadsheetBench solver mode: SkillOpt's ``multi`` codegen agent as a graph.

SkillOpt's SpreadsheetBench config (configs/spreadsheetbench/default.yaml) runs mode
``multi``: up to 30 model calls per item, each reply's first
fenced block run against the input, an execution error sent back as the next user
message, and -- in training rollouts only -- a wrong output told which cells are wrong
(`ops`, `refactory_dsh.feedback`). Each call is one ``claude -p`` whose prompt is the
whole conversation flattened into a ``History:`` block (`ops.frame`) and whose system
file is ``codegen_system.md`` plus the ``## Skill`` section, appended to Claude Code's own
system prompt (``--append-system-prompt-file``). SkillOpt sends no tool schemas of its
own, but each ``claude -p`` carried Claude Code's default tools, run under ``dontAsk`` in
a fresh empty directory (also its ``--add-dir``): a command, a write or a read elsewhere
was refused with Claude Code's "Permission to use ... has been denied" result (60% of
so-s43's 3,217 target calls had at least one: Bash 2,500, Write 109). So the solve node
declares ``tool_policy="refused"`` (`refactory_dsh.tool_policy`): the same tools, refused
the same way, in an empty directory of its own. With no tools at all (``"none"``, before)
the model wrote its tool calls and their results as text, inventing the output, in about
a third of its turns.

Run settings, as SkillOpt's runs used them, are not part of the graph: the
``claude_code`` worker, ``claude-opus-4-6`` at effort ``medium``, permission mode
``dontAsk`` (which ``refused`` requires). A refused step loads no settings whatever the
route names (SkillOpt's ``project`` found none in its empty directory). Its Claude Code
is pinned to SkillOpt's, 2.1.299 (`Solver.claude_code_versions`): a run whose solve step
would run another (the SDK's bundled 2.1.281, say) is refused before it starts, so name
it with ``--claude-executable``.

**Deviations from SkillOpt as shipped**, each recorded where it is decided:

1. *No item budget, no per-call wall*: SkillOpt gives an item 600 s (``exec_timeout``)
   and each call 120 s. Here a solve call ends on inactivity only
   (``REFACTORY_MODEL_INACTIVITY_S``: nothing streamed and no command running), and the
   one limit is the harness backstop (``REFACTORY_MODEL_BACKSTOP_S``, 4 h by default),
   for each call and for each item's turn loop. An item past it is the model's own
   failure (its record's ``error_kind`` is ``backstop``: scored 0, never rerun or left
   out of the score). The lab's
   SkillOpt wrapper is to run with the same backstop as its item limit.
2. *Interpreter*: a pinned environment with SkillOpt's versions (`interpreter`), with
   pandas installed in it rather than added through ``PYTHONPATH``.
3. *Grader*: ``recalc`` by default (`recalc`); ``SB_GRADER=cached`` is SkillOpt's.
   Scores compare only with a SkillOpt run graded the same way (the lab's
   ``SB_GRADER=recalc`` wrapper), never with SkillOpt's cached-grader numbers. A
   recalculation that fails is a harness incident (`recalc.RecalcFailed`), not a wrong
   answer.
4. *Sandbox*: model code cannot read the golden file, the dataset or the run
   (`sandbox`); SkillOpt's could. Stricter, so it can only lower a score that came
   from reading the answer (none of so-s42's generated code mentions the golden
   file, ``listdir`` or ``glob``; port spec §3).
5. *Tools*: refused, as SkillOpt's were (above), except that a tool which needs no
   permission finds an empty directory where SkillOpt's found its system-prompt file, and
   a write to Claude Code's memory folder succeeds for the session (SkillOpt's home was
   read-only, so it failed there); the folder goes with the session.
6. *Train score*: a train item its gold feedback told was wrong scores 0, however its
   later turns end (`refactory_dsh.feedback.SCORING`); SkillOpt scores the attempt after
   the feedback, which passed nearly every clean train item and left the search nothing
   to learn from. The conversation, the feedback included, is unchanged.
7. *Formula note*: under ``recalc`` the verification report (the analysts' view) says the
   grader recalculates formulas and that the ``=`` cells it lists hold text
   (`skillopt_port.FORMULA_NOTE_RECALC`), not SkillOpt's "openpyxl cannot evaluate them",
   which is false there and led a proposer to argue for plain values; ``cached`` keeps it.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from refactory_dsh import nn
from refactory_dsh.feedback import CALL as FEEDBACK_CALL

from . import skillopt_port as port
from .tasks import INPUT

OPS = "refactory_packages.spreadsheetbench.ops"
TASK = "refactory_packages.spreadsheetbench.tasks:SpreadsheetBenchTask"
#: The ops that open the item's input (`RunContext.item_file`), so declare it in reads.
INPUT_READERS = (f"{OPS}:prepare", f"{OPS}:run_turn", f"{OPS}:finalize")

#: The Claude Code SkillOpt's runs used (the lab's ``claude --version``): its system
#: prompt and tool text are what the solve call is compared with.
SKILLOPT_CLAUDE_CODE = "2.1.299"
#: SkillOpt's ``max_turns`` (configs/spreadsheetbench/default.yaml:31).
MAX_TURNS = 30
#: The solve node's system prompt: the system file, then the skill section.
SYSTEM_PROMPT = port.CODEGEN_SYSTEM + "{{skill_section}}"
SYSTEM_PROMPT_KNOB = "solve__system_prompt"
#: A crashed solve call is run again in a fresh session up to 4 times, 5 attempts in
#: all, as SkillOpt's wrapper attempts each call (rather than the data pass's default 2).
SOLVE_RETRY = {"on": ["crash"], "times": 4}


class Solver(nn.Mode):
    """SkillOpt's SpreadsheetBench target agent over the Task's items."""

    name = "solver"
    description = "solve a SpreadsheetBench item with SkillOpt's multi-turn codegen loop"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob(port.INITIAL_SKILL, "str",
                             description="skill appended to the solver's system prompt"),
            "max_completion_tokens": nn.Knob(
                0, "int", lo=0, frozen=True,
                description="output-token cap on each solve call; 0 = no cap (SkillOpt's "
                            "claude backend drops its 16384)"),
            "data_parallelism": nn.Knob(24, "int", lo=1, hi=64, frozen=True,
                                        description="concurrent items (SkillOpt's workers)"),
            "max_turns": nn.Knob(MAX_TURNS, "int", lo=1, hi=MAX_TURNS, frozen=True,
                                 description="model calls per item"),
            "turn_pause_s": nn.Knob(3, "int", lo=0, hi=3, frozen=True,
                                    description="cooldown after each model call"),
            SYSTEM_PROMPT_KNOB: nn.Knob(SYSTEM_PROMPT, "str", frozen=True,
                                        description="the solve call's system prompt "
                                                    "(frozen; use the skill knob)"),
        }

    def build(self) -> nn.Graph:
        g = nn.Graph()
        g.add(nn.Node("data", "data", subgraph_entry="prepare", subgraph_exit="finalize",
                      task_ref=TASK, writes=("current_item.json", INPUT),
                      produces=("item_scores",),
                      parallelism_knob="data_parallelism", knobs=("data_parallelism",)))
        g.add(nn.Node("prepare", "op", call=f"{OPS}:prepare",
                      reads=("current_item.json", INPUT),
                      knobs=("skill",),
                      writes=("history.json", "turn.json", "target_system_prompt.txt",
                              "target_user_prompt.txt", "spreadsheet_preview.txt"),
                      produces=("skill_section", "has_case"),
                      knob_outputs={"skill_section": ("skill",), "has_case": (),
                                    "target_system_prompt.txt": ("skill",),
                                    "history.json": (), "turn.json": (),
                                    "target_user_prompt.txt": (), "spreadsheet_preview.txt": ()},
                      notes="record the item's start and build SkillOpt's first message"))
        g.add(nn.Node("has_case", "gate", evaluator=f"{OPS}:has_case", inputs=("has_case",),
                      notes="an item with no test case skips the loop"))
        g.add(nn.Node("frame", "op", call=f"{OPS}:frame", reads=("history.json",),
                      produces=("solver_prompt",),
                      notes="flatten the conversation into the History: prompt"))
        g.add(nn.Node("solve", "agent", prompt_template="{{solver_prompt}}",
                      system_prompt_template=SYSTEM_PROMPT, writes=("reply.txt",),
                      tool_policy="refused", role="solver", outputs_only=True,
                      retry=SOLVE_RETRY,
                      knobs=("max_completion_tokens", SYSTEM_PROMPT_KNOB),
                      field_bindings=(("system_prompt_template", SYSTEM_PROMPT_KNOB),)))
        g.add(nn.Node("run_turn", "op", call=f"{OPS}:run_turn",
                      reads=("reply.txt", "history.json", "turn.json", "current_item.json",
                             INPUT),
                      knobs=("turn_pause_s",),
                      writes=("history.json", "turn.json", "code.py", "raw.txt", "pred.xlsx"),
                      produces=("turn_status",),
                      notes="run the reply's code; an error is the next user message"))
        g.add(nn.Node("gold_check", "op", call=FEEDBACK_CALL, produces=("train_feedback",),
                      notes="train rollouts only: is the clean run's output right"))
        g.add(nn.Node("next_turn", "op", call=f"{OPS}:next_turn",
                      reads=("history.json", "turn.json"), inputs=("train_feedback",),
                      writes=("history.json", "turn.json"), produces=("turn_status",),
                      notes="a clean run ends the item unless train feedback says it is wrong"))
        g.add(nn.Node("turn_gate", "gate", evaluator=f"{OPS}:turn_gate", reads=("turn.json",),
                      knobs=("max_turns",), writes=("turn.json",),
                      max_iterations=MAX_TURNS - 1, on_exhausted="continue",
                      notes="another turn while turns and budget remain"))
        g.add(nn.Node("finalize", "op", call=f"{OPS}:finalize",
                      reads=("current_item.json", "turn.json", "history.json", "code.py",
                             INPUT),
                      knobs=("max_turns",),
                      writes=("final.json", "conversation.json", "pred.xlsx"),
                      produces=("finished",),
                      notes="the item's outcome before grading"))
        g.connect("prepare", "has_case")
        g.connect("has_case", "frame", condition="proceed")
        g.connect("has_case", "finalize", condition="skip")
        g.connect("frame", "solve")
        g.connect("solve", "run_turn")
        g.connect("run_turn", "gold_check")
        g.connect("gold_check", "next_turn")
        g.connect("next_turn", "turn_gate")
        g.connect("turn_gate", "frame", condition="reloop")
        g.connect("turn_gate", "finalize", condition="proceed")
        return g

    @classmethod
    def upgrade_state(cls, state: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """A state saved before the input was an item file (`tasks.INPUT`): one whose data
        node for this Task (protected, so no edit changes it) does not declare the input in
        its writes. Then each such data node declares it, and each node running one of the
        ops that open it (`ops.prepare`, ``run_turn``, ``finalize``, by call, whatever its
        id) declares it in its reads, as this version's graph does. A state that already
        declares it -- any state this version saved -- is returned as it is, with no note:
        never a change to anything an edit chose."""
        nodes = ((state.get("graph") or {}).get("nodes") or {}) if isinstance(state, Mapping) \
            else {}
        data = [n for n in nodes.values() if n.get("kind") == "data" and n.get("task_ref") == TASK]
        if not data or all(INPUT in (n.get("writes") or ()) for n in data):
            return dict(state), []
        state = copy.deepcopy(dict(state))
        nodes = state["graph"]["nodes"]
        notes: list[str] = []
        for node_id, node in sorted(nodes.items()):
            if node.get("kind") == "data" and node.get("task_ref") == TASK:
                key = "writes"
            elif node.get("call") in INPUT_READERS:
                key = "reads"
            else:
                continue
            declared = list(node.get(key) or [])
            if INPUT not in declared:
                node[key] = [*declared, INPUT]
                notes.append(f"{node_id} now declares {INPUT} in its {key} (a state saved "
                             "before the input was an item file)")
        return state, notes

    def claude_code_versions(self) -> dict[str, str]:
        """The solve step runs on SkillOpt's Claude Code (`SKILLOPT_CLAUDE_CODE`)."""
        return {"solve": SKILLOPT_CLAUDE_CODE}

    def protected_nodes(self) -> tuple[str, ...]:
        """The data pass, the train-feedback node (V17's channel) and the item's outcome."""
        return ("data", "gold_check", "finalize")

    def run(self, workdir: Path, *, model: Any = None, task: Mapping[str, Any] | None = None,
            memory: Any = None, **kwargs: Any) -> nn.RunResult:
        """Put a nonzero ``max_completion_tokens`` on the solve node's route, as LiveMath
        does (`refactory_packages.livemath.mode.Solver.run`)."""
        cap = int(self.values().get("max_completion_tokens") or 0)
        if cap > 0 and model is not None and callable(getattr(model, "for_node", None)):
            routes = dict(getattr(model, "_node_routes", None) or {})
            routes["solve"] = {**routes.get("solve", {}), "max_completion_tokens": str(cap)}
            model = copy.copy(model)
            model._node_routes = routes
        return super().run(workdir, model=model, task=task, memory=memory, **kwargs)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0

    def guidance(self) -> str:
        return ("The skill is appended to the solver's system prompt for every item; the "
                "graph can be edited too -- the solve loop and the steps before it (only "
                "data, gold_check and finalize are protected). Keep every change general -- "
                "it must help on items no rollout saw.")
