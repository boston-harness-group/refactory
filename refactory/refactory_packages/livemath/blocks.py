"""Solver block library for the structure ablation grid.

Each builder returns a ``BlockResult``: the nodes and edges that replace
LiveMath's single ``solve`` node inside the per-item data pass.  Every block
shares ``format_prompt`` as its entry and writes ``answer.txt`` from its exit
node, graded by the unchanged grader.

All agent nodes use the same ``SYSTEM_PROMPT`` (with ``{{skill_section}}``),
``tool_policy="none"``, and ``timeout_s=SOLVE_TIMEOUT_S``, so the skill, tool
policy and route settings reach every solving / judging / verifying agent
identically.
"""
from __future__ import annotations

from collections import namedtuple

from refactory_dsh import nn

from .mode import OPS, SOLVE_TIMEOUT_S, SYSTEM_PROMPT

BLOCK_OPS = "refactory_packages.livemath.block_ops"

BlockResult = namedtuple(
    "BlockResult", ["nodes", "edges", "entry", "exit", "agent_node_ids"],
)

# ── prompt constants ──────────────────────────────────────────────────────

VERIFY_PROMPT_TEMPLATE = (
    "The following question was answered by a solver. Review the answer and "
    "provide the correct final choice.\n"
    "\n"
    "{{user_prompt}}\n"
    "\n"
    "## Solver's Answer\n"
    "{{draft_response}}\n"
    "\n"
    "## Your Task\n"
    "Carefully verify the solver's reasoning. If the answer is correct, "
    "confirm it. If not, provide the correct answer. Output only the final "
    "choice label inside <answer>...</answer> tags."
)

JUDGE_PROMPT_TEMPLATE = (
    "Two independent solvers answered the following question. Read both "
    "attempts and pick the best final answer.\n"
    "\n"
    "{{user_prompt}}\n"
    "\n"
    "## Attempt A\n"
    "{{attempt_a_text}}\n"
    "\n"
    "## Attempt B\n"
    "{{attempt_b_text}}\n"
    "\n"
    "## Your Task\n"
    "Compare both attempts. Pick the answer with the strongest reasoning. "
    "Output only the final choice label inside <answer>...</answer> tags."
)

# Mirrors SkillOpt's chat-backend refinement text (multi-turn, max_turns 2).
SELF_CHECK_PROMPT = (
    "Your previous answer was:\n"
    "{{draft_response}}\n"
    "\n"
    "Re-evaluate the exact option wording. If needed, correct it. "
    "Output only the final choice label inside <answer>...</answer>."
)


# ── block builders ────────────────────────────────────────────────────────


def _format_prompt_node() -> nn.Node:
    """The shared format_prompt node, identical across all blocks."""
    return nn.Node(
        "format_prompt", "op",
        call=f"{OPS}:format_prompt",
        reads=("current_item.json",),
        knobs=("skill", "use_theorem", "use_sketch"),
        produces=("user_prompt", "skill_section"),
    )


def build_baseline() -> BlockResult:
    """Block (a): the current single solve — no structural change."""
    fmt = _format_prompt_node()
    solve = nn.Node(
        "solve", "agent",
        prompt_template="{{user_prompt}}",
        system_prompt_template=SYSTEM_PROMPT,
        writes=("answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    return BlockResult(
        nodes=[fmt, solve],
        edges=[("format_prompt", "solve")],
        entry="format_prompt",
        exit="solve",
        agent_node_ids=["solve"],
    )


def build_solve_then_verify() -> BlockResult:
    """Block (b): solve → read draft → verify."""
    fmt = _format_prompt_node()
    solve_draft = nn.Node(
        "solve_draft", "agent",
        prompt_template="{{user_prompt}}",
        system_prompt_template=SYSTEM_PROMPT,
        writes=("draft_answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    read_draft = nn.Node(
        "read_draft", "op",
        call=f"{BLOCK_OPS}:read_draft",
        reads=("draft_answer.txt",),
        produces=("draft_response",),
    )
    verify = nn.Node(
        "verify", "agent",
        prompt_template=VERIFY_PROMPT_TEMPLATE,
        system_prompt_template=SYSTEM_PROMPT,
        writes=("answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    return BlockResult(
        nodes=[fmt, solve_draft, read_draft, verify],
        edges=[
            ("format_prompt", "solve_draft"),
            ("solve_draft", "read_draft"),
            ("read_draft", "verify"),
        ],
        entry="format_prompt",
        exit="verify",
        agent_node_ids=["solve_draft", "verify"],
    )


def build_two_attempts_then_judge() -> BlockResult:
    """Block (c): fork → [solve_a | solve_b] → join → merge → judge."""
    fmt = _format_prompt_node()
    attempt_fork = nn.Node(
        "attempt_fork", "fork",
        targets=("solve_a", "solve_b"),
    )
    solve_a = nn.Node(
        "solve_a", "agent",
        prompt_template="{{user_prompt}}",
        system_prompt_template=SYSTEM_PROMPT,
        writes=("attempt_a.txt",),
        produces=("answer_a",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    solve_b = nn.Node(
        "solve_b", "agent",
        prompt_template="{{user_prompt}}",
        system_prompt_template=SYSTEM_PROMPT,
        writes=("attempt_b.txt",),
        produces=("answer_b",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    attempt_join = nn.Node(
        "attempt_join", "join",
        sources=("solve_a", "solve_b"),
    )
    merge = nn.Node(
        "merge_attempts", "op",
        call=f"{BLOCK_OPS}:merge_attempts",
        reads=("attempt_a.txt", "attempt_b.txt"),
        produces=("attempt_a_text", "attempt_b_text"),
    )
    judge = nn.Node(
        "judge", "agent",
        prompt_template=JUDGE_PROMPT_TEMPLATE,
        system_prompt_template=SYSTEM_PROMPT,
        writes=("answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    return BlockResult(
        nodes=[fmt, attempt_fork, solve_a, solve_b, attempt_join, merge, judge],
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
    )


def build_solve_then_self_check() -> BlockResult:
    """Block (d): solve → read draft → self-check."""
    fmt = _format_prompt_node()
    solve_draft = nn.Node(
        "solve_draft", "agent",
        prompt_template="{{user_prompt}}",
        system_prompt_template=SYSTEM_PROMPT,
        writes=("draft_answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    read_draft = nn.Node(
        "read_draft", "op",
        call=f"{BLOCK_OPS}:read_draft",
        reads=("draft_answer.txt",),
        produces=("draft_response",),
    )
    self_check = nn.Node(
        "self_check", "agent",
        prompt_template=SELF_CHECK_PROMPT,
        system_prompt_template=SYSTEM_PROMPT,
        writes=("answer.txt",),
        tool_policy="none",
        role="solver",
        timeout_s=SOLVE_TIMEOUT_S,
        knobs=("max_completion_tokens",),
    )
    return BlockResult(
        nodes=[fmt, solve_draft, read_draft, self_check],
        edges=[
            ("format_prompt", "solve_draft"),
            ("solve_draft", "read_draft"),
            ("read_draft", "self_check"),
        ],
        entry="format_prompt",
        exit="self_check",
        agent_node_ids=["solve_draft", "self_check"],
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
