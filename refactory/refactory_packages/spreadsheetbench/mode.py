"""SpreadsheetBench solver mode: SkillOpt's ``multi`` codegen agent as a graph.

SkillOpt's SpreadsheetBench config (configs/spreadsheetbench/default.yaml) runs mode
``multi``: up to 30 model calls per item, each reply's first
fenced block run against the input, an execution error sent back as the next user
message, and -- in training rollouts only -- a wrong output told which cells are wrong
(`ops`, `refactory_dsh.feedback`). Each call is one ``claude -p`` whose prompt is the
whole conversation flattened into a ``History:`` block (`ops.frame`) and whose system
file is ``codegen_system.md`` plus the ``## Skill`` section. The solve node has no tools
(``tool_policy="none"``): SkillOpt sends no tool schemas, though its CLI calls carried
Claude Code's default tools, which ``dontAsk`` denied (port spec §1.5, §7 Q2).

Run settings, as SkillOpt's runs used them, are not part of the graph: the
``claude_code`` worker, ``claude-opus-4-6`` at effort ``medium``, permission mode
``dontAsk``, setting sources ``project``.

**Deviations from SkillOpt as shipped**, each recorded where it is decided:

1. *No item budget, no per-call wall*: SkillOpt gives an item 600 s (``exec_timeout``)
   and each call 120 s. Here a solve call ends on inactivity only
   (``REFACTORY_MODEL_INACTIVITY_S``: nothing streamed and no command running), and the
   one limit is the harness backstop (``REFACTORY_MODEL_BACKSTOP_S``, 4 h by default),
   for each call and for each item's turn loop. An item past it is a harness incident
   (its record's ``error_kind`` is ``harness_incident``), never a score of 0. The lab's
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
5. *Tools*: none on the solve call (above).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from refactory_dsh import nn
from refactory_dsh.feedback import CALL as FEEDBACK_CALL

from . import skillopt_port as port

OPS = "refactory_packages.spreadsheetbench.ops"
TASK = "refactory_packages.spreadsheetbench.tasks:SpreadsheetBenchTask"

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
                      task_ref=TASK, writes=("current_item.json",), produces=("item_scores",),
                      parallelism_knob="data_parallelism", knobs=("data_parallelism",)))
        g.add(nn.Node("prepare", "op", call=f"{OPS}:prepare", reads=("current_item.json",),
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
                      tool_policy="none", role="solver", outputs_only=True,
                      retry=SOLVE_RETRY,
                      knobs=("max_completion_tokens", SYSTEM_PROMPT_KNOB),
                      field_bindings=(("system_prompt_template", SYSTEM_PROMPT_KNOB),)))
        g.add(nn.Node("run_turn", "op", call=f"{OPS}:run_turn",
                      reads=("reply.txt", "history.json", "turn.json", "current_item.json"),
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
                      reads=("current_item.json", "turn.json", "history.json", "code.py"),
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
        return ("The skill is the only lever: it is appended to the solver's system prompt "
                "for every item. Keep it general -- it must help on items no rollout saw.")
