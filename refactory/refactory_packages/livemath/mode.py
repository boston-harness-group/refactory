"""LiveMathematicianBench one-node solver mode.

The solver graph: data → [format_prompt → solve] → finish.

System prompt ported from SkillOpt (MIT) —
skillopt/envs/livemathematicianbench/prompts/rollout_system.md.
SkillOpt solver defaults:
  exec_timeout = 1800 s -- applied as the solve node's ``timeout_s``
    (``Solver.solve_timeout_s``). SkillOpt's LiveMath config says 300
    (configs/livemathematicianbench/default.yaml:17), but arm A's runs override it
    with ``--cfg-options env.exec_timeout=1800`` (SkillOpt's run-train.sh:14 and
    run-train-seed.sh:20, recorded in each run's out/config.json ``exec_timeout``),
    and the outer-loop protocol has arm B match arm A.
  max_completion_tokens = 0 (no cap) -- the ``max_completion_tokens`` knob,
    applied to the solve node's route only (``Solver.run``). SkillOpt's LiveMath
    config says 16384 (livemathematicianbench/default.yaml:16), but arm A's
    ``claude_chat`` backend drops it: ``_call_messages`` starts with
    ``del max_completion_tokens`` (skillopt/model/claude_backend.py:310), so A's
    solver output is never capped -- 73 of the 1,280 calls in A's seed-42 run
    (runs/skillopt-train-20261023-e10-s42) produced more than 16,384 output
    tokens, the largest 103,088. ``--knob max_completion_tokens=N`` sets a cap.
  reasoning_effort = medium (_base_/default.yaml:10) -- an agent node carries no
    per-node effort; it is a run setting (``--effort medium``, or ``node_routes``
    for ``solve``), not part of this graph.
  model = gpt-5.5 (_base_/default.yaml:7) -- likewise a run setting; arm A ran
    claude-opus-4-6 (its out/config.json ``target_model``).

Arm-A parity. Arm A's solver is ``claude -p`` (claude_backend.py ``_run_claude_print``)
with only Claude Code's project settings loaded, effort medium and 24 workers.
SkillOpt's own default is ``user,project`` (claude_backend.py:19), but arm A's runs
set ``CLAUDE_SETTING_SOURCES=project`` (SkillOpt's run-train-seed.sh:17): every solver
and optimizer call in A's runs/skillopt-train-20261023-e10-s<seed>/calls.jsonl
records ``setting_sources`` ``project``. Loading ``user`` as well puts the host
account's own CLAUDE.md and settings into every call. One known difference remains:
A's solver carries Claude Code's default tools (the ``dontAsk`` mode refuses them;
2-4% of A's solves spend a turn on a refused call), while this solve node runs with
no tools. The arm-B parity command, as the 2026-11-05 v2 seeds ran it (from
main after #583; A's ``skills/skill_v0000.md`` is the starting skill)::

    refactory optimize livemath/solver \
        --strategy skillopt --runs-root <runs> \
        --worker-backend claude_code --permission-mode dontAsk --model claude-opus-4-6 --effort medium \
        --setting-sources project \
        --knob "skill=$(cat <A out>/skills/skill_v0000.md)" --knob data_parallelism=24 \
        --allowed-ops set_knob \
        --parts-manifest <parts.json> --holdout 0 --seed <seed> --epochs 10 \
        --batch-size 1 --optimizer random --max-elapsed-s 172800 --max-evaluations 500 \
        --strategy-option max_epochs=10 --strategy-option seed=<seed> \
        --strategy-option early_stop_patience=3 --strategy-option early_stop_at_full_val=true \
        --strategy-option edit_budget_max=4 --strategy-option edit_budget_min=2 \
        --strategy-option analyst_edit_budget=4 --strategy-option claude_backend_wrapper=true \
        --strategy-option slow_update_samples=20 --strategy-option group_size=8 \
        --strategy-option merge_batch_size=8 --verbose

(the v2 seeds also passed ``--allowed-ops text_patch``, which has since been folded
into ``set_knob``; the strategy options are A's out/config.json settings, and
``claude_backend_wrapper`` frames every optimizer call's user message as A's Claude
backend does -- ``History:\n- User: ...`` then ``Answer the latest user request.``
(skillopt/claude_backend.py); off by default, ``=false`` or omitting it sends the
bare message. The solve node always sends its question framed that way
(``ops.claude_backend_prompt``). The SkillOpt strategy evaluates on the manifest's
test part after training, as A does). With that command, seeds 42-45, the best
skills scored on the same solver over the 125 test items: B 88.2/125 (0.705, 3
scores per seed) against A's best skills 88.3/125 (0.706, 2 scores per seed).
Earlier, on the ``user,project`` route: with A's best seed-42 skill, this solver
scored 57.6% on the plain DSH API route and 69.6% on ``claude_code``, against A's
72.8%; A's four best skills averaged 72.0% on ``claude_code`` against A's 73.2%.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar, Mapping

from refactory_dsh import nn

OPS = "refactory_packages.livemath.ops"
TASK = "refactory_packages.livemath.tasks:LiveMathTask"

# Ported from skillopt/envs/livemathematicianbench/prompts/rollout_system.md (MIT)
# {skill_section} replaced by {{skill_section}} for template rendering. The file
# ends with a newline, which SkillOpt's _build_system (rollout.py:16-21) keeps.
SYSTEM_PROMPT = (
    "You are an expert mathematical reasoning agent solving multiple-choice questions.\n"
    "\n"
    "{{skill_section}}## Task Format\n"
    "You will receive one mathematics multiple-choice question and its answer choices.\n"
    "Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.\n"
    "\n"
    "## Answer Format\n"
    "Think step by step, then provide your final answer inside <answer>...</answer> tags.\n"
    "Inside the tags, output only the single choice label, such as A or C.\n"
    "\n"
    "Example:\n"
    "<answer>B</answer>\n"
)

# Arm A's exec_timeout (run-train.sh:14, run-train-seed.sh:20); SkillOpt's own
# configs/livemathematicianbench/default.yaml:17 default is 300.
SOLVE_TIMEOUT_S = 1800
# Arm A's solver output cap: none. SkillOpt's LiveMath config sets 16384
# (configs/livemathematicianbench/default.yaml:16), but the claude_chat backend arm
# A runs on discards it (skillopt/model/claude_backend.py:310). 0 = no cap.
MAX_COMPLETION_TOKENS = 0


class Solver(nn.Mode):
    """One-agent-node solver for LiveMathematicianBench items.

    The ``block`` constructor parameter selects the solver structure:

    * ``"baseline"`` (default) — the current single ``solve`` node; its lowered
      graph is pinned by the lowering snapshot.
    * ``"solve_then_verify"`` — solve, read draft, verify.
    * ``"two_attempts_then_judge"`` — fork into two solves, join, judge.
    * ``"solve_then_self_check"`` — solve, read draft, self-check.

    ``block`` is a construction-time choice, not a knob — the optimizer cannot
    search over it.
    """

    name = "solver"
    description = "solve a LiveMathematicianBench item with a single agent call"
    #: The solve node's wall limit; a subclass (or a graph edit of ``timeout_s``)
    #: changes it.
    solve_timeout_s: ClassVar[int] = SOLVE_TIMEOUT_S

    def __init__(
        self,
        block: str = "baseline",
        state: dict[str, Any] | None = None,
    ) -> None:
        from .blocks import BLOCK_NAMES

        if block not in BLOCK_NAMES:
            raise ValueError(
                f"unknown block {block!r}; known: {list(BLOCK_NAMES)}"
            )
        self._block = block
        super().__init__(state=state)

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "skill": nn.Knob("", "str", description="skill prompt injected into the system message"),
            # SkillOpt configs/livemathematicianbench/default.yaml:21-22
            "use_theorem": nn.Knob(False, "bool",
                                   description="include the item's theorem as a ## Theorem section"),
            "use_sketch": nn.Knob(False, "bool",
                                   description="include the item's proof sketch as a ## Proof Sketch section"),
            # Uncapped, as arm A's solver is (``MAX_COMPLETION_TOKENS``). A run
            # setting, not a search target: frozen against the optimizer, set
            # with ``--knob max_completion_tokens=N``.
            "max_completion_tokens": nn.Knob(
                MAX_COMPLETION_TOKENS, "int", lo=0, frozen=True,
                description="output-token cap on each solve-node model call; 0 = no cap"),
            # Concurrent data-node items. A resource setting, not a search
            # target: frozen against the optimizer, set with
            # ``--knob data_parallelism=N``.
            "data_parallelism": nn.Knob(
                1, "int", lo=1, hi=64, frozen=True,
                description="concurrent data-node items"),
        }

    def _declared_entries(self) -> dict[str, dict[str, Any]]:
        """The mode's knobs plus the selected block's frozen system-prompt knobs."""
        from .blocks import BLOCKS

        entries = super()._declared_entries()
        block = BLOCKS[getattr(self, "_block", "baseline")](self.solve_timeout_s)
        for name, knob in block.knobs.items():
            entries[name] = {"value": knob.default, "spec": knob.spec()}
        return entries

    def build(self) -> nn.Graph:
        """The data pass over the selected block, then ``finish``.

        Every block, the baseline included, comes from ``blocks.BLOCKS``; the
        baseline's lowered graph is pinned by its lowering snapshot
        (tests/lowering_snapshots/livemath-solver.graph.json). Every agent node
        gets ``solve_timeout_s``.
        """
        from .blocks import BLOCKS

        result = BLOCKS[getattr(self, "_block", "baseline")](self.solve_timeout_s)
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data",
            subgraph_entry=result.entry,
            subgraph_exit=result.exit,
            task_ref=TASK,
            writes=("current_item.json",),
            produces=("item_scores",),
            parallelism_knob="data_parallelism",
            knobs=("data_parallelism",),
        ))
        for node in result.nodes:
            graph.add(node)
        for source, target in result.edges:
            graph.connect(source, target)
        graph.add(nn.Node(
            "finish", "op",
            call=f"{OPS}:finish",
            writes=("result.json",),
            produces=("finished",),
        ))
        graph.connect("data", "finish")
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
        """Put the ``max_completion_tokens`` knob on every agent node's route.

        The graph has no per-node route field, so the cap travels the way any
        per-node route setting does: as entries in the model's ``node_routes``.
        ``for_node(id)`` then carries it in the worker's requested route, and
        the worker observer sets it as ``maxTokens`` on the DSH config.
        """
        cap = int(self.values().get("max_completion_tokens") or 0)
        # Every agent node of the graph this mode holds, so a clone
        # (``type(self)(state=...)``, which does not know the block name) caps them too.
        agent_ids = [node.id for node in self.graph().nodes.values() if node.kind == "agent"]
        if cap > 0 and model is not None and callable(getattr(model, "for_node", None)):
            routes = dict(getattr(model, "_node_routes", None) or {})
            for node_id in agent_ids:
                routes[node_id] = {
                    **routes.get(node_id, {}),
                    "max_completion_tokens": str(cap),
                }
            model = copy.copy(model)
            model._node_routes = routes
        return super().run(workdir, model=model, task=task, memory=memory, **kwargs)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class SolverVerify(Solver):
    """Solve-then-verify block as its own mode for the structure grid."""

    name = "solver_verify"
    description = "solve then verify with a second agent"

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        super().__init__(block="solve_then_verify", state=state)


class SolverJudge(Solver):
    """Two-attempts-then-judge block as its own mode for the structure grid."""

    name = "solver_judge"
    description = "fork two solve attempts then judge with a third agent"

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        super().__init__(block="two_attempts_then_judge", state=state)


class SolverSelfCheck(Solver):
    """Solve-then-self-check block as its own mode for the structure grid."""

    name = "solver_self_check"
    description = "solve then self-check with SkillOpt-style refinement"

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        super().__init__(block="solve_then_self_check", state=state)
