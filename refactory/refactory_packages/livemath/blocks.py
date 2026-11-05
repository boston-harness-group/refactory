"""Solver block library for the structure ablation grid.

Each builder returns a ``BlockResult``: the nodes and edges that replace
LiveMath's single ``solve`` node inside the per-item data pass.  Every block
shares ``format_prompt`` as its entry and writes ``answer.txt`` from its exit
node, graded by the unchanged grader.

Every agent node is one ``claude -p`` call in arm A's backend, so every agent
node sends its whole user message framed as that backend frames it
(``ops.claude_backend_prompt``: ``History:\\n- User: ...`` then ``Answer the
latest user request.``).  The solving calls send ``{{solver_prompt}}`` --
``format_prompt``'s framed question, byte for byte the baseline's solve prompt.
The verifier, judge and self-check send a prompt an op builds from the bare
question (``user_prompt``) and the earlier answers, then frames
(``block_ops``).

All agent nodes use the same ``SYSTEM_PROMPT`` (with ``{{skill_section}}``),
``tool_policy="none"`` and the mode's ``solve_timeout_s``, so the skill, tool
policy and route settings reach every solving / judging / verifying agent
identically.
"""
from __future__ import annotations

from collections import namedtuple

from refactory_dsh import nn

from .mode import OPS, SOLVE_TIMEOUT_S, SYSTEM_PROMPT

BLOCK_OPS = "refactory_packages.livemath.block_ops"

BlockResult = namedtuple(
    "BlockResult", ["nodes", "edges", "entry", "exit", "agent_node_ids", "knobs"],
)


def system_prompt_knob(node_id: str) -> str:
    """The name of an agent node's system-prompt knob (``Mode`` auto-declares it)."""
    return f"{node_id}__system_prompt"


def frozen_system_prompt_knobs(agent_node_ids) -> dict[str, nn.Knob]:
    """One frozen system-prompt knob per agent node of a block.

    Every agent node's system prompt is a knob by default (F14). LiveMath freezes
    each one, so grid cells differ only in structure and the optimizer edits only
    the ``skill`` knob. A block declares exactly its own agent nodes' knobs, so
    none is declared without a consumer.
    """
    return {
        system_prompt_knob(node_id): nn.Knob(
            SYSTEM_PROMPT, "str", frozen=True,
            description=f"system prompt for {node_id} (frozen; use the skill knob)")
        for node_id in agent_node_ids
    }


def _bound_system_prompt(node_id: str) -> dict:
    """The knobs and field binding that tie an agent node's system prompt to its knob."""
    name = system_prompt_knob(node_id)
    return {
        "knobs": ("max_completion_tokens", name),
        "field_bindings": (("system_prompt_template", name),),
    }


# ── node helpers ──────────────────────────────────────────────────────────


def _format_prompt_node() -> nn.Node:
    """The shared format_prompt node, identical across all blocks."""
    return nn.Node(
        "format_prompt", "op",
        call=f"{OPS}:format_prompt",
        reads=("current_item.json",),
        knobs=("skill", "use_theorem", "use_sketch"),
        produces=("user_prompt", "solver_prompt", "skill_section"),
    )


def _agent(node_id: str, prompt_template: str, writes: str, timeout_s: int,
           produces: tuple[str, ...] = ()) -> nn.Node:
    """One solving / verifying / judging call, configured as the baseline solve."""
    return nn.Node(
        node_id, "agent",
        prompt_template=prompt_template,
        system_prompt_template=SYSTEM_PROMPT,
        writes=(writes,),
        **({"produces": produces} if produces else {}),
        tool_policy="none",
        role="solver",
        timeout_s=timeout_s,
        **_bound_system_prompt(node_id),
    )


def _prompt_op(node_id: str, fn: str, reads: tuple[str, ...], produces: str) -> nn.Node:
    """An op that builds the next agent's framed prompt from the earlier answers."""
    return nn.Node(
        node_id, "op",
        call=f"{BLOCK_OPS}:{fn}",
        reads=reads,
        produces=(produces,),
    )


# ── block builders ────────────────────────────────────────────────────────


def build_baseline(timeout_s: int = SOLVE_TIMEOUT_S) -> BlockResult:
    """Block (a): the current single solve -- no structural change."""
    fmt = _format_prompt_node()
    solve = nn.Node(
        "solve", "agent",
        # The question framed as arm A's Claude backend sends it
        # (``ops.claude_backend_prompt``); ``user_prompt`` stays the bare question.
        prompt_template="{{solver_prompt}}",
        system_prompt_template=SYSTEM_PROMPT,
        writes=("answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=timeout_s,
        **_bound_system_prompt("solve"),
    )
    return BlockResult(
        nodes=[fmt, solve],
        edges=[("format_prompt", "solve")],
        entry="format_prompt",
        exit="solve",
        agent_node_ids=["solve"],
        knobs=frozen_system_prompt_knobs(["solve"]),
    )


def build_solve_then_verify(timeout_s: int = SOLVE_TIMEOUT_S) -> BlockResult:
    """Block (b): solve → build the verify prompt → verify."""
    return BlockResult(
        nodes=[
            _format_prompt_node(),
            _agent("solve_draft", "{{solver_prompt}}", "draft_answer.txt", timeout_s),
            _prompt_op("read_draft", "verify_prompt", ("draft_answer.txt",), "verify_prompt"),
            _agent("verify", "{{verify_prompt}}", "answer.txt", timeout_s),
        ],
        edges=[
            ("format_prompt", "solve_draft"),
            ("solve_draft", "read_draft"),
            ("read_draft", "verify"),
        ],
        entry="format_prompt",
        exit="verify",
        agent_node_ids=["solve_draft", "verify"],
        knobs=frozen_system_prompt_knobs(["solve_draft", "verify"]),
    )


def build_two_attempts_then_judge(timeout_s: int = SOLVE_TIMEOUT_S) -> BlockResult:
    """Block (c): fork → [solve_a | solve_b] → join → build the judge prompt → judge."""
    return BlockResult(
        nodes=[
            _format_prompt_node(),
            nn.Node("attempt_fork", "fork", targets=("solve_a", "solve_b")),
            _agent("solve_a", "{{solver_prompt}}", "attempt_a.txt", timeout_s,
                   produces=("answer_a",)),
            _agent("solve_b", "{{solver_prompt}}", "attempt_b.txt", timeout_s,
                   produces=("answer_b",)),
            nn.Node("attempt_join", "join", sources=("solve_a", "solve_b")),
            _prompt_op("merge_attempts", "judge_prompt",
                       ("attempt_a.txt", "attempt_b.txt"), "judge_prompt"),
            _agent("judge", "{{judge_prompt}}", "answer.txt", timeout_s),
        ],
        edges=[
            ("format_prompt", "attempt_fork"),
            ("attempt_fork", "attempt_join"),
            ("solve_a", "attempt_join"),
            ("solve_b", "attempt_join"),
            ("attempt_join", "merge_attempts"),
            ("merge_attempts", "judge"),
        ],
        entry="format_prompt",
        exit="judge",
        agent_node_ids=["solve_a", "solve_b", "judge"],
        knobs=frozen_system_prompt_knobs(["solve_a", "solve_b", "judge"]),
    )


def build_solve_then_self_check(timeout_s: int = SOLVE_TIMEOUT_S) -> BlockResult:
    """Block (d): solve → build the refinement prompt → self-check."""
    return BlockResult(
        nodes=[
            _format_prompt_node(),
            _agent("solve_draft", "{{solver_prompt}}", "draft_answer.txt", timeout_s),
            _prompt_op("read_draft", "self_check_prompt", ("draft_answer.txt",),
                       "self_check_prompt"),
            _agent("self_check", "{{self_check_prompt}}", "answer.txt", timeout_s),
        ],
        edges=[
            ("format_prompt", "solve_draft"),
            ("solve_draft", "read_draft"),
            ("read_draft", "self_check"),
        ],
        entry="format_prompt",
        exit="self_check",
        agent_node_ids=["solve_draft", "self_check"],
        knobs=frozen_system_prompt_knobs(["solve_draft", "self_check"]),
    )


# ── registry ──────────────────────────────────────────────────────────────

BLOCK_NAMES = (
    "baseline",
    "solve_then_verify",
    "two_attempts_then_judge",
    "solve_then_self_check",
)

BLOCKS: dict[str, object] = {
    "baseline": build_baseline,
    "solve_then_verify": build_solve_then_verify,
    "two_attempts_then_judge": build_two_attempts_then_judge,
    "solve_then_self_check": build_solve_then_self_check,
}
