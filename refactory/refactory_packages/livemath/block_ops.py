"""Operations that build the second-stage prompts of the non-baseline blocks.

Each op reads the bare question (``user_prompt``, from ``format_prompt``) and the
earlier agents' answers, writes the next agent's whole user message, and frames it
as arm A's Claude backend frames every call (``ops.claude_backend_prompt``), the
same framing the solving calls get through ``solver_prompt``.
"""
from __future__ import annotations

from typing import Any

from refactory_dsh import nn

from .ops import claude_backend_prompt, declared_io as _declared

VERIFY_HEAD = (
    "The following question was answered by a solver. Review the answer and "
    "provide the correct final choice."
)
VERIFY_TASK = (
    "## Your Task\n"
    "Carefully verify the solver's reasoning. If the answer is correct, "
    "confirm it. If not, provide the correct answer. Output only the final "
    "choice label inside <answer>...</answer> tags."
)
JUDGE_HEAD = (
    "Two independent solvers answered the following question. Read both "
    "attempts and pick the best final answer."
)
JUDGE_TASK = (
    "## Your Task\n"
    "Compare both attempts. Pick the answer with the strongest reasoning. "
    "Output only the final choice label inside <answer>...</answer> tags."
)
# SkillOpt's refinement turn on its exec (Codex/Claude Code harness) path:
# skillopt/envs/livemathematicianbench/rollout.py:88-95 appends this to the full
# ``_build_user`` message, and the exec prompt ends with the answer-format line at
# rollout.py:102. (The chat path at rollout.py:230-234 sends the previous answer
# without the question, which would leave the self-check unable to re-read the
# options.)
SELF_CHECK_INSTRUCTION = "Re-evaluate the exact option wording. If needed, correct it."
SELF_CHECK_FORMAT = "Output only the final choice label inside <answer>...</answer>."


def _question(ctx: Any) -> str:
    return str(ctx.facts.get("user_prompt") or "")


def verify_message(question: str, draft: str) -> str:
    """The verifier's user message, before framing."""
    return (f"{VERIFY_HEAD}\n\n{question}\n\n## Solver's Answer\n{draft}\n\n"
            f"{VERIFY_TASK}")


def judge_message(question: str, attempt_a: str, attempt_b: str) -> str:
    """The judge's user message, before framing."""
    return (f"{JUDGE_HEAD}\n\n{question}\n\n## Attempt A\n{attempt_a}\n\n"
            f"## Attempt B\n{attempt_b}\n\n{JUDGE_TASK}")


def self_check_message(question: str, draft: str) -> str:
    """The self-check's user message, before framing (rollout.py:88-95, :102)."""
    return (f"{question}\n\n## Previous Attempt\n{draft}\n\n"
            f"{SELF_CHECK_INSTRUCTION}\n\n{SELF_CHECK_FORMAT}")


@nn.by_declaration(files=("draft_answer.txt",))
def verify_prompt(ctx: Any) -> dict[str, Any]:
    """The verify node's framed prompt: question and the draft (its one declared read,
    ``draft_answer.txt`` as built), as its one declared fact."""
    reads, facts = _declared(ctx, verify_prompt.declared_files, ["verify_prompt"])
    message = verify_message(_question(ctx), ctx.read_text(reads[0] or "draft_answer.txt"))
    return {facts[0]: claude_backend_prompt(message)}


#: What the judge is shown for a fork attempt that gave no answer (it crashed through
#: its retries): the other attempt still reaches it.
NO_ATTEMPT = "(This attempt gave no answer: its session did not answer.)"


@nn.by_declaration(files=("attempt_a.txt", "attempt_b.txt"))
def judge_prompt(ctx: Any) -> dict[str, Any]:
    """The judge node's framed prompt: question and both fork branches' answers (its two
    declared reads, in order), as its one declared fact.

    A branch that did not answer is shown as `NO_ATTEMPT`; with neither there is nothing
    to judge, and the missing attempts are the error (the item has no answer)."""
    reads, facts = _declared(ctx, judge_prompt.declared_files, ["judge_prompt"])
    names = [name for name in reads if name]
    if not any(ctx.exists(name) for name in names):
        raise FileNotFoundError(f"neither attempt was written: {', '.join(names)}")
    attempts = [ctx.read_text(name) if name and ctx.exists(name) else NO_ATTEMPT
                for name in reads]
    message = judge_message(_question(ctx), *attempts)
    return {facts[0]: claude_backend_prompt(message)}


@nn.by_declaration(files=("draft_answer.txt",))
def self_check_prompt(ctx: Any) -> dict[str, Any]:
    """The self-check node's framed prompt: question, choices and the draft (its one
    declared read), as its one declared fact."""
    reads, facts = _declared(ctx, self_check_prompt.declared_files, ["self_check_prompt"])
    message = self_check_message(_question(ctx), ctx.read_text(reads[0] or "draft_answer.txt"))
    return {facts[0]: claude_backend_prompt(message)}
