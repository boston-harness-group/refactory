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


# Ported from skillopt/envs/livemathematicianbench/rollout.py:16-21 and
# rollout.py:31-53 (MIT)
def format_prompt(ctx: Any) -> dict[str, Any]:
    """Build the user prompt and skill section for the solver node."""
    item = ctx.read_json("current_item.json", {}) or {}

    # Build user prompt — rollout.py:31-53
    question = item.get("question", "")
    choices = item.get("choices", [])
    parts = [
        f"## Question\n{question}",
        f"## Choices\n{_format_choices(choices)}",
    ]
    # Gated by the use_theorem / use_sketch knobs exactly as SkillOpt's
    # _build_user() at rollout.py:41-44; both default False per
    # configs/livemathematicianbench/default.yaml:21-22
    if ctx.knob("use_theorem", False) and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if ctx.knob("use_sketch", False) and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
    user_prompt = "\n\n".join(parts)

    # Build skill section — rollout.py:16-21
    skill = str(ctx.knob("skill", "")).strip()
    if skill:
        skill_section = f"## Skill\n{skill}\n\n"
    else:
        skill_section = ""

    return {"user_prompt": user_prompt, "skill_section": skill_section}


def finish(ctx: Any) -> dict[str, Any]:
    return {"finished": True}
