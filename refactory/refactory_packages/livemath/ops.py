"""LiveMathematicianBench operations.

Prompt formatting ported from SkillOpt (MIT) —
skillopt/envs/livemathematicianbench/rollout.py:16-53.
"""
from __future__ import annotations

from typing import Any


# Ported from skillopt/envs/livemathematicianbench/rollout.py:24-28 (MIT)
def _format_choices(choices: list[dict]) -> str:
    return "\n".join(
        f"{choice['label']}. {choice['text']}"
        for choice in choices
    )


# Ported from skillopt/envs/livemathematicianbench/rollout.py:16-21 (MIT)
def skill_section(skill: Any) -> str:
    """The ``## Skill`` block ``_build_system`` puts in the system prompt."""
    text = str(skill).strip()
    return f"## Skill\n{text}\n\n" if text else ""


# Ported from skillopt/envs/livemathematicianbench/rollout.py:31-53 (MIT)
def build_user_prompt(item: dict, *, use_theorem: bool = False, use_sketch: bool = False) -> str:
    """The solver's user prompt for one item (``_build_user``)."""
    question = item.get("question", "")
    choices = item.get("choices", [])
    parts = [
        f"## Question\n{question}",
        f"## Choices\n{_format_choices(choices)}",
    ]
    # Gated by the use_theorem / use_sketch knobs exactly as SkillOpt's
    # _build_user() at rollout.py:41-44; both default False per
    # configs/livemathematicianbench/default.yaml:21-22
    if use_theorem and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if use_sketch and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
    return "\n\n".join(parts)


def format_prompt(ctx: Any) -> dict[str, Any]:
    """Build the user prompt and skill section for the solver node."""
    item = ctx.read_json("current_item.json", {}) or {}
    return {
        "user_prompt": build_user_prompt(
            item,
            use_theorem=bool(ctx.knob("use_theorem", False)),
            use_sketch=bool(ctx.knob("use_sketch", False)),
        ),
        "skill_section": skill_section(ctx.knob("skill", "")),
    }


def finish(ctx: Any) -> dict[str, Any]:
    return {"finished": True}
