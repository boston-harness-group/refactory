"""LiveMath skill graph: arm A's seed skill as an explicit four-step graph.

The seed skill SkillOpt starts LiveMath from (arm A's ``skill_v0000.md``) is one
free-form procedure in four sections: "Option Comparison", "Theorem-Level
Precision", "Hypotheses" and "Final Answer". This mode runs each section as its
own agent step inside LiveMath's per-item data pass::

    data → [prepare → compare → notes_prompts → step_fork
              → (precision | hypotheses) → step_join
              → decide_prompt → decide → extract_answer] → finish

* ``prepare`` (op) formats the item exactly as ``livemath/solver``'s
  ``format_prompt`` does (the task shuffles the choices per item and seed) and
  turns each step's section knob into that step's ``## Skill`` block.
* ``compare`` (agent) analyses every option and writes notes.
* ``precision`` and ``hypotheses`` (agents) run in parallel on the question and
  compare's notes, each writing notes.
* ``decide`` (agent) reads the question and all three notes and answers in the
  baseline solver's format; it writes ``answer.txt``, which LiveMath's unchanged
  grader scores.
* ``extract_answer`` (op) records the option label the grader reads from that
  reply (``grading.parse_choice_label``) and whether it is a valid choice.

Every agent node is configured as the baseline ``solve`` node: LiveMath's solver
system prompt (its skill block holding only that step's section), ``tool_policy``
none, ``role`` solver, ``timeout_s`` = ``solve_timeout_s``, the
``max_completion_tokens`` cap (none by default) on its route, and its whole user
message framed as arm A's Claude backend frames every call
(``ops.claude_backend_prompt``). The prompt builders between steps are ops.

Knobs: one open section knob per step (``compare_skill``, ``precision_skill``,
``hypotheses_skill``, ``decide_skill``; defaults are the seed skill's sections),
plus ``use_theorem`` / ``use_sketch`` as in the baseline. Frozen, as the baseline
freezes them: each agent's ``<node>__system_prompt`` knob and the
``max_completion_tokens`` / ``data_parallelism`` run settings. There is no
``skill`` knob.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar, Mapping

from refactory_dsh import nn

from .grading import normalize_label, parse_choice_label
from .mode import MAX_COMPLETION_TOKENS, OPS, SOLVE_TIMEOUT_S, SYSTEM_PROMPT, TASK
from .ops import build_user_prompt, claude_backend_prompt, skill_section

SKILL_GRAPH_OPS = "refactory_packages.livemath.skill_graph"

#: The four agent steps, in order.
STEPS = ("compare", "precision", "hypotheses", "decide")

# The seed skill's four sections, verbatim from arm A's skill_v0000.md (SkillOpt's
# LiveMath initial skill; also in tests/fixtures/skillopt_arm_a_optimizer_prompts.json).
# Each keeps its own ``##`` heading; the skill's title line is not part of any section.
SEED_SECTIONS: dict[str, str] = {
    "compare": (
        "## Option Comparison\n"
        "- Compare all options before committing. The correct choice is often the "
        "strongest statement justified by the question, while nearby distractors are "
        "weaker, overstrong, or miss an equality case.\n"
        "- Track exact quantifiers such as \"there exists\", \"for every\", "
        "\"if and only if\", and \"exactly when\"."
    ),
    "precision": (
        "## Theorem-Level Precision\n"
        "- Check whether an option weakens the conclusion by dropping a "
        "characterization, equality clause, or full equivalence.\n"
        "- Check whether an option overstates the theorem by upgrading regularity, "
        "removing scale restrictions, or changing an existential statement into a "
        "universal one."
    ),
    "hypotheses": (
        "## Hypotheses\n"
        "- Verify the hypotheses and domain carefully. Distractors often keep the "
        "theorem shape but alter the required assumptions.\n"
        "- Pay close attention to equality cases, extremal conditions, and whether a "
        "result applies to the full family or only a restricted subfamily."
    ),
    "decide": (
        "## Final Answer\n"
        "- Output the final answer as the single option label only."
    ),
}

#: Each step's notes file, and the heading its notes go under in later prompts.
NOTES = {
    "compare": ("compare_notes.txt", "## Option Comparison Notes"),
    "precision": ("precision_notes.txt", "## Theorem-Level Precision Notes"),
    "hypotheses": ("hypotheses_notes.txt", "## Hypotheses Notes"),
}

COMPARE_TASK = (
    "## Your Task\n"
    "This is the first of several analysis steps. Analyse every option in turn: "
    "decide which statement is the strongest one the question justifies, and check "
    "each option's exact quantifiers against it. Write your analysis as notes for "
    "the next steps. A final answer is not required."
)
PRECISION_TASK = (
    "## Your Task\n"
    "Using the option comparison notes above, check each option's theorem-level "
    "precision: whether it weakens or overstates what can be proven. Write your "
    "analysis as notes for the final step. A final answer is not required."
)
HYPOTHESES_TASK = (
    "## Your Task\n"
    "Using the option comparison notes above, check each option's hypotheses, "
    "domain, equality cases and scope. Write your analysis as notes for the final "
    "step. A final answer is not required."
)
DECIDE_TASK = (
    "## Your Task\n"
    "Weigh the notes above against the question and its options, and choose the "
    "single correct option. Think step by step, then provide your final answer "
    "inside <answer>...</answer> tags, with only the single choice label inside."
)


def skill_knob(step: str) -> str:
    """The name of a step's section knob."""
    return f"{step}_skill"


def skill_section_fact(step: str) -> str:
    """The fact ``prepare`` writes a step's ``## Skill`` block to."""
    return f"{step}_skill_section"


def system_prompt_knob(node_id: str) -> str:
    """An agent node's system-prompt knob (the name ``Mode`` auto-declares)."""
    return f"{node_id}__system_prompt"


def step_system_prompt(step: str) -> str:
    """LiveMath's solver system prompt with this step's skill block in it."""
    return SYSTEM_PROMPT.replace("{{skill_section}}", "{{" + skill_section_fact(step) + "}}")


# ── messages (before framing) ─────────────────────────────────────────────


def _notes_block(step: str, notes: str) -> str:
    return f"{NOTES[step][1]}\n{notes.strip()}"


def compare_message(question: str) -> str:
    return f"{question}\n\n{COMPARE_TASK}"


def precision_message(question: str, compare_notes: str) -> str:
    return f"{question}\n\n{_notes_block('compare', compare_notes)}\n\n{PRECISION_TASK}"


def hypotheses_message(question: str, compare_notes: str) -> str:
    return f"{question}\n\n{_notes_block('compare', compare_notes)}\n\n{HYPOTHESES_TASK}"


def decide_message(question: str, compare_notes: str, precision_notes: str,
                   hypotheses_notes: str) -> str:
    return (f"{question}\n\n{_notes_block('compare', compare_notes)}\n\n"
            f"{_notes_block('precision', precision_notes)}\n\n"
            f"{_notes_block('hypotheses', hypotheses_notes)}\n\n{DECIDE_TASK}")


# ── ops ───────────────────────────────────────────────────────────────────


def prepare(ctx: Any) -> dict[str, Any]:
    """The item's question and choices, compare's framed prompt, every step's skill block.

    ``user_prompt`` is ``livemath/solver``'s ``format_prompt`` message for the same
    item and knobs, byte for byte (``ops.build_user_prompt``).
    """
    item = ctx.read_json("current_item.json", {}) or {}
    user_prompt = build_user_prompt(
        item,
        use_theorem=bool(ctx.knob("use_theorem", False)),
        use_sketch=bool(ctx.knob("use_sketch", False)),
    )
    facts = {
        "user_prompt": user_prompt,
        "compare_prompt": claude_backend_prompt(compare_message(user_prompt)),
    }
    for step in STEPS:
        facts[skill_section_fact(step)] = skill_section(ctx.knob(skill_knob(step), ""))
    return facts


def _question(ctx: Any) -> str:
    return str(ctx.facts.get("user_prompt") or "")


def notes_prompts(ctx: Any) -> dict[str, Any]:
    """precision's and hypotheses' framed prompts: the question and compare's notes."""
    notes = ctx.read_text(NOTES["compare"][0])
    return {
        "precision_prompt": claude_backend_prompt(precision_message(_question(ctx), notes)),
        "hypotheses_prompt": claude_backend_prompt(hypotheses_message(_question(ctx), notes)),
    }


def decide_prompt(ctx: Any) -> dict[str, Any]:
    """decide's framed prompt: the question and all three steps' notes."""
    message = decide_message(_question(ctx), *(ctx.read_text(NOTES[s][0]) for s in
                                               ("compare", "precision", "hypotheses")))
    return {"decide_prompt": claude_backend_prompt(message)}


def extract_answer(ctx: Any) -> dict[str, Any]:
    """The option label LiveMath's grader reads from decide's reply, and whether it is valid.

    Same rules as the grader (``grading.parse_choice_label``); ``answer.txt`` is left
    as decide wrote it, so the task grades decide's reply exactly as it grades the
    baseline solve's.
    """
    item = ctx.read_json("current_item.json", {}) or {}
    choices = item.get("choices", [])
    label = parse_choice_label(ctx.read_text("answer.txt"), choices)
    valid = {normalize_label(choice.get("label", "")) for choice in choices}
    return {"answer_label": label, "answer_valid": label in valid}


# ── graph ─────────────────────────────────────────────────────────────────


def _op(node_id: str, fn: str, *, reads: tuple[str, ...], produces: tuple[str, ...],
        knobs: tuple[str, ...] = ()) -> nn.Node:
    return nn.Node(node_id, "op", call=f"{SKILL_GRAPH_OPS}:{fn}", reads=reads,
                   produces=produces, **({"knobs": knobs} if knobs else {}))


def _agent(step: str, writes: str, timeout_s: int) -> nn.Node:
    """One step, configured as the baseline solve node."""
    knob = system_prompt_knob(step)
    return nn.Node(
        step, "agent",
        prompt_template="{{" + step + "_prompt}}",
        system_prompt_template=step_system_prompt(step),
        writes=(writes,),
        tool_policy="none",
        role="solver",
        timeout_s=timeout_s,
        knobs=("max_completion_tokens", knob),
        field_bindings=(("system_prompt_template", knob),),
    )


class SkillGraph(nn.Mode):
    """Arm A's seed skill as an explicit compare → (precision | hypotheses) → decide graph."""

    name = "skill_graph"
    description = ("solve a LiveMathematicianBench item with the seed skill's four "
                   "sections as four agent steps")
    #: Every agent node's wall limit, as ``Solver.solve_timeout_s``.
    solve_timeout_s: ClassVar[int] = SOLVE_TIMEOUT_S

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        knobs: dict[str, nn.Knob] = {
            skill_knob(step): nn.Knob(
                SEED_SECTIONS[step], "str",
                description=f"skill section in the {step} step's system message")
            for step in STEPS
        }
        knobs.update({
            "use_theorem": nn.Knob(False, "bool",
                                   description="include the item's theorem as a ## Theorem section"),
            "use_sketch": nn.Knob(False, "bool",
                                  description="include the item's proof sketch as a ## Proof Sketch section"),
            # Run settings, frozen as in ``Solver``.
            "max_completion_tokens": nn.Knob(
                MAX_COMPLETION_TOKENS, "int", lo=0, frozen=True,
                description="output-token cap on each agent call; 0 = no cap"),
            "data_parallelism": nn.Knob(
                1, "int", lo=1, hi=64, frozen=True,
                description="concurrent data-node items"),
        })
        # Frozen as the baseline freezes ``solve__system_prompt``: the steps' section
        # knobs are what the optimizer edits.
        for step in STEPS:
            knobs[system_prompt_knob(step)] = nn.Knob(
                step_system_prompt(step), "str", frozen=True,
                description=f"system prompt for {step} (frozen; use {skill_knob(step)})")
        return knobs

    def build(self) -> nn.Graph:
        t = self.solve_timeout_s
        compare_notes, precision_notes, hypotheses_notes = (NOTES[s][0] for s in
                                                            ("compare", "precision", "hypotheses"))
        nodes = [
            nn.Node(
                "data", "data",
                subgraph_entry="prepare",
                subgraph_exit="extract_answer",
                task_ref=TASK,
                writes=("current_item.json",),
                produces=("item_scores",),
                parallelism_knob="data_parallelism",
                knobs=("data_parallelism",),
            ),
            _op("prepare", "prepare", reads=("current_item.json",),
                produces=("user_prompt", "compare_prompt",
                          *(skill_section_fact(s) for s in STEPS)),
                knobs=(*(skill_knob(s) for s in STEPS), "use_theorem", "use_sketch")),
            _agent("compare", compare_notes, t),
            _op("notes_prompts", "notes_prompts", reads=(compare_notes,),
                produces=("precision_prompt", "hypotheses_prompt")),
            nn.Node("step_fork", "fork", targets=("precision", "hypotheses")),
            _agent("precision", precision_notes, t),
            _agent("hypotheses", hypotheses_notes, t),
            nn.Node("step_join", "join", sources=("precision", "hypotheses")),
            _op("decide_prompt", "decide_prompt",
                reads=(compare_notes, precision_notes, hypotheses_notes),
                produces=("decide_prompt",)),
            _agent("decide", "answer.txt", t),
            _op("extract_answer", "extract_answer", reads=("answer.txt", "current_item.json"),
                produces=("answer_label", "answer_valid")),
            nn.Node("finish", "op", call=f"{OPS}:finish", writes=("result.json",),
                    produces=("finished",)),
        ]
        graph = nn.Graph()
        for node in nodes:
            graph.add(node)
        for source, target in (
            ("prepare", "compare"),
            ("compare", "notes_prompts"),
            ("notes_prompts", "step_fork"),
            ("step_fork", "step_join"),
            ("precision", "step_join"),
            ("hypotheses", "step_join"),
            ("step_join", "decide_prompt"),
            ("decide_prompt", "decide"),
            ("decide", "extract_answer"),
            ("data", "finish"),
        ):
            graph.connect(source, target)
        return graph

    def run(
        self,
        workdir: Path,
        *,
        model: Any = None,
        task: Mapping[str, Any] | None = None,
        memory: Any = None,
        **kwargs: Any,
    ) -> nn.RunResult:
        """Put the ``max_completion_tokens`` cap on every agent node's route, as ``Solver.run``."""
        cap = int(self.values().get("max_completion_tokens") or 0)
        if cap > 0 and model is not None and callable(getattr(model, "for_node", None)):
            routes = dict(getattr(model, "_node_routes", None) or {})
            for node in self.graph().nodes.values():
                if node.kind == "agent":
                    routes[node.id] = {**routes.get(node.id, {}),
                                       "max_completion_tokens": str(cap)}
            model = copy.copy(model)
            model._node_routes = routes
        return super().run(workdir, model=model, task=task, memory=memory, **kwargs)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0
