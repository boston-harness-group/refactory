"""All prompt templates for SkillOpt, each citing the SkillOpt source file and line range.

Constants ported from the ``skillopt/prompts/`` directory (microsoft/SkillOpt, MIT).
"""

from __future__ import annotations

import json

# ── Markers (skill.py:14-15) ───────────────────────────────────────────────
SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

# ── Analyst prompts (prompts/analyst_error.md, prompts/analyst_success.md) ─
FAILURE_ANALYST_SYSTEM = """\
You are an expert failure-analysis agent for AI agent tasks.

You will be given MULTIPLE failed agent trajectories from a single minibatch
and the current skill document.
Your job is to identify the most important COMMON failure patterns across
the batch and propose a concise set of skill edits.

## Analysis Process
1. Read ALL trajectories in the minibatch.
2. Identify the most prevalent, systematic failure patterns across them.
3. For each pattern, classify its failure type.
4. Propose skill edits that address the COMMON patterns — not individual edge cases.
5. Edits must be generalizable; do not hardcode task-specific values.
6. Only patch gaps in the skill — do not duplicate existing content.

You will be told the maximum number of edits (the budget L). Produce AT MOST L edits,
focusing on the highest-impact patterns. You may produce fewer if warranted.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{
  "batch_size": <number of trajectories analysed>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
  "patch": {
    "reasoning": "<why these edits address the batch's common failures>",
    "edits": [
      {"op": "append",       "content": "<markdown to add at end of skill>"},
      {"op": "insert_after", "target": "<exact heading/text to insert after>", "content": "<markdown>"},
      {"op": "replace",      "target": "<exact text to replace>",              "content": "<replacement>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}
Only include edits that are needed. "edits" can be an empty list if no patch is warranted.

IMPORTANT: The skill document may contain a section between
<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
This is a PROTECTED section managed by a separate slow-update process.
Do NOT propose any edits that target, modify, or delete content within
these markers.\
"""
# Source: skillopt/prompts/analyst_error.md (lines 1-41)

SUCCESS_ANALYST_SYSTEM = """\
You are an expert success-pattern analyst for AI agents.

You will be given MULTIPLE successful agent trajectories from a single minibatch
and the current skill document. Your job is to identify generalizable behavior
patterns that are COMMON across the batch and worth encoding in the skill.

## Rules
- Only propose patches for patterns NOT already covered in the skill.
- Focus on patterns that appear across MULTIPLE trajectories in the batch.
- Be concise. Patterns must generalize beyond specific tasks.
- Prefer reinforcing existing sections over adding new top-level sections.

You will be told the maximum number of edits (the budget L). Produce AT MOST L edits,
focusing on the most broadly applicable patterns. You may produce fewer if warranted.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number of trajectories analysed>,
  "success_patterns": ["<pattern 1>", "<pattern 2>"],
  "patch": {
    "reasoning": "<why these patterns are worth encoding>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}
"edits" may be empty if the skill already covers all observed patterns.

IMPORTANT: The skill document may contain a section between
<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
This is a PROTECTED section managed by a separate slow-update process.
Do NOT propose any edits that target, modify, or delete content within
these markers.\
"""
# Source: skillopt/prompts/analyst_success.md (lines 1-36)

# ── Merge prompts (prompts/merge_failure.md, merge_success.md, merge_final.md) ─
MERGE_FAILURE_SYSTEM = """\
You are a skill-edit coordinator. You receive multiple independently-proposed patches
from FAILURE analysis of agent trajectories. Merge them into ONE coherent, non-redundant patch.

Merge guidelines:
1. **Deduplicate**: keep the best-worded version of similar edits.
2. **Resolve conflicts**: if patches contradict on the same point,
   choose the one with stronger justification or synthesize both.
3. **Preserve unique insights**: include all non-redundant corrective edits.
4. **Prevalent-pattern bias**: edits appearing consistently across multiple patches
   address systematic failures — preserve them with HIGH priority.
   Edits from only one patch may be discarded if task-specific.
5. **Independence**: no two edits in the merged patch may target the same text region.
6. **Support count**: for each merged edit, estimate how many source patches support it.
7. **PROTECTED SECTION**: The skill may contain a section between
   <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
   Do NOT merge or produce any edits that target content within these markers.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary of key consolidation decisions>",
    "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if insert_after or replace or delete>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "failure"
    }
  ]
}\
"""
# Source: skillopt/prompts/merge_failure.md (lines 1-30)

MERGE_SUCCESS_SYSTEM = """\
You are a skill-edit coordinator. You receive multiple independently-proposed patches
from SUCCESS analysis of agent trajectories. Merge them into ONE coherent patch
that reinforces effective patterns.

Merge guidelines:
1. **Deduplicate**: keep only the most generalizable version of similar patterns.
2. **Be conservative**: success-driven patches reinforce existing behavior.
   Only include edits for patterns NOT already in the skill.
3. **Prevalent-pattern bias**: patterns seen across many successful trajectories
   are most worth encoding.
4. **Support count**: estimate how many source patches support each merged edit.
5. **PROTECTED SECTION**: The skill may contain a section between
   <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
   Do NOT merge or produce any edits that target content within these markers.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "success"
    }
  ]
}\
"""
# Source: skillopt/prompts/merge_success.md (lines 1-28)

MERGE_FINAL_SYSTEM = """\
You are a skill-edit coordinator performing the FINAL merge. You receive two
pre-merged patch groups:
1. **Failure-driven patches** (corrective, high priority)
2. **Success-driven patches** (reinforcement, lower priority)

Merge guidelines:
1. **FAILURE PATCHES TAKE PRIORITY**: the primary goal of skill reflection is to
   fix failures. Failure-driven edits should be preserved unless they directly
   conflict with a well-supported success pattern.
2. **Deduplicate**: if a failure edit and success edit cover the same point,
   keep the failure version.
3. **Preserve success insights**: include success edits that cover patterns
   NOT addressed by failure edits.
4. **Higher-level merges represent broader consensus**: edits that survived
   previous merge rounds (higher level) should be given priority.
5. **Carry forward support_count and source_type for each edit.**
6. **PROTECTED SECTION**: The skill may contain a section between
   <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
   Do NOT merge or produce any edits that target content within these markers.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary of priority decisions>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "failure|success"
    }
  ]
}\
"""
# Source: skillopt/prompts/merge_final.md (lines 1-33)

# ── Ranking prompt (prompts/ranking.md) ────────────────────────────────────
RANK_SYSTEM = """\
You are an expert skill-optimization optimizer. You receive a skill document and a pool
of proposed edits. Your job is to RANK the edits by importance and select the top ones.

Ranking criteria (in order of priority):
1. **Systematic impact**: edits that address widespread, recurring failure patterns
   across many tasks should rank highest. A rule that fixes 50%% of failures beats
   one that fixes a single edge case.
2. **Complementarity**: edits that fill gaps in the current skill (not duplicate
   existing content) rank higher.
3. **Generality**: edits phrased as general principles rank higher than those
   tied to specific question types or entities.
4. **Actionability**: edits with clear, concrete guidance rank higher than vague advice.

You will be told how many edits to select (the budget).

Respond ONLY with a valid JSON object:
{
  "reasoning": "<brief justification for your ranking decisions>",
  "selected_indices": [<0-based indices of the top edits, in priority order>]
}\
"""
# Source: skillopt/prompts/ranking.md (lines 1-20)

# ── Slow update prompt (prompts/slow_update.md) ───────────────────────────
SLOW_UPDATE_SYSTEM = """\
You are a strategic skill advisor for an AI agent optimization system.

Your role is different from the per-step analyst. The per-step analyst sees
individual trajectories and proposes local patches. YOU see how the skill has
evolved across an entire epoch by comparing the SAME tasks under two consecutive
skill versions. This longitudinal view lets you identify systemic drift,
regressions, and persistent blind spots that step-level edits cannot catch.

## What You Receive

1. **Previous epoch's skill** and **current epoch's skill** — to see what changed.
2. **Longitudinal comparison** — the same 20 training tasks rolled out under
   both skills, categorized into: regressions, persistent failures,
   improvements, and stable successes.
3. **Previous slow update guidance** (if any) — the guidance you (or a prior
   invocation of you) wrote at the end of the last epoch. This guidance was
   active during the current epoch's step-level optimization. You must evaluate
   whether it helped or hurt based on the longitudinal comparison results.

## Your Process

1. **Reflect on the previous guidance** (if provided):
   - Which parts of the previous guidance were effective? (Evidence: tasks that
     improved or stayed correct.)
   - Which parts failed or backfired? (Evidence: regressions or persistent
     failures that the guidance was supposed to address.)
   - Were there blind spots the previous guidance missed entirely?
   Include this reflection in your "reasoning" field.

2. **Write updated guidance** that:
   - Retains and strengthens parts of the previous guidance that proved effective.
   - Revises or removes parts that were ineffective or counterproductive.
   - Adds new instructions to address newly observed regressions and persistent
     failures.

## Output Requirements

Write a **strategic guidance block** that will OVERWRITE the previous guidance
in the protected section of the skill document. This section is READ-ONLY to
all subsequent step-level optimization — only you can overwrite it at the next
epoch boundary.

Your guidance must:
- Be written as **direct, actionable instructions** to the target model
  (the AI agent that will read and follow the skill).
- Focus on helping the target get problems RIGHT — not on analysis or
  explanation of what went wrong.
- Prioritize: (1) preventing regressions, (2) fixing persistent failures,
  (3) reinforcing successful patterns.
- Be concise but comprehensive — you have no length limit, but every sentence
  should earn its place.
- NOT duplicate content already in the main skill body — complement it.
- Address the target directly (e.g., "When you encounter X, always do Y"
  rather than "The agent should...").

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{
  "reasoning": "<your reflection on the previous guidance AND analysis of the longitudinal comparison>",
  "slow_update_content": "<the exact guidance text to insert into the protected section>"
}\
"""
# Source: skillopt/prompts/slow_update.md (lines 1-60)

# ── Meta skill prompt (prompts/meta_skill.md) ─────────────────────────────
META_SKILL_SYSTEM = """\
You are a optimizer-coach for an AI agent skill optimization system.

Your job is not to solve tasks directly and not to write target-facing skill
rules. Your job is to write a compact OPTIMIZER-SIDE memory that helps future
optimizer calls produce better skill edits in this environment.

## What You Receive

1. The previous epoch's last-step skill.
2. The current epoch's last-step skill.
3. A longitudinal comparison on the SAME sampled tasks under those two skills.
4. The previous optimizer meta skill, if one existed.

## Your Goal

Write a concise meta skill that improves future optimizer behavior in stages such
as failure analysis, success analysis, patch merging, and edit ranking.

This meta skill should capture things like:
- Which kinds of edits tend to help in this environment.
- Which kinds of edits tend to be too vague, redundant, brittle, or harmful.
- What level of abstraction works best for rules here.
- What failure-repair patterns should be prioritized.
- What regression risks future optimizer calls should guard against.

## Important Constraints

- Address the FUTURE OPTIMIZER directly, not the target.
- Focus on how to write better edits and organize better skill updates.
- Use evidence from the adjacent-epoch comparison, not generic advice.
- Keep it compact and high-signal. Prefer a few durable principles.
- Revise or remove parts of the previous meta skill if they did not help.
- Do not output target-facing task instructions.
- Do not restate the whole skill; summarize editing strategy.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<brief reflection on what editing directions helped or hurt>",
  "meta_skill_content": "<compact optimizer-side guidance for future edit generation and selection>"
}\
"""
# Source: skillopt/prompts/meta_skill.md (lines 1-40)




# ── Format helpers ─────────────────────────────────────────────────────────
#
# Every user message below reproduces SkillOpt's text byte for byte: section
# headings, blank lines, group lines and where the optimizer meta skill goes.
# Trajectories are SkillOpt's own formatter over the rollout results the
# mode's package renders from F6 item records (``env.py``).


def format_meta_skill_context(meta_skill_text: str) -> str:
    """Render the optimizer meta skill as a prompt block — meta_skill.py:18-30."""
    content = (meta_skill_text or "").strip()
    if not content:
        return ""
    return (
        "## Optimizer Meta Skill\n"
        "This is optimizer-side memory distilled from prior epoch transitions in "
        "this environment. Use it to improve how you propose, merge, and rank "
        "skill edits. Prefer it when the current evidence is ambiguous, but do "
        "not force it if the current trajectories clearly contradict it.\n\n"
        f"{content}"
    )


def _text(value: object) -> str:
    """reflect.py:54-62 ``_clip_text``: render optional fields in full."""
    if value is None:
        return ""
    return str(value)


def record_fail_reason(record: dict) -> str:
    """The record's fail reason; an errored item's reason is its error."""
    reason = record.get("fail_reason")
    if reason:
        return _text(reason)
    if record.get("error_text"):
        kind = record.get("error_kind") or "error"
        return f"error: {kind}: {record['error_text']}"
    return ""


def fmt_trajectory(conversation: list) -> str:
    """Port of SkillOpt's ``fmt_trajectory`` (skillopt/gradient/reflect.py:74-115).

    One line per conversation entry: tool calls, step records, ``system``
    entries as ``[verification]`` and anything else as ``[<role>]`` with
    ``role`` defaulting to ``agent``.
    """
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {_text(item)}")
            continue
        if item.get("type") == "tool_call":
            cmd = _text(item.get("cmd"))
            obs = _text(item.get("obs"))
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = _text(item.get("reasoning"))
            action = _text(item.get("action"))
            feedback = _text(item.get("env_feedback"))
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif item.get("role") == "system":
            msg = _text(item.get("content"))
            lines.append(f"[verification] {msg}")
        else:
            msg = _text(item.get("content"))
            role = item.get("role", "agent")
            lines.append(f"[{role}] {msg}")
    return "\n".join(lines)


def fmt_minibatch_trajectories(results: list[dict]) -> str:
    """Port of SkillOpt's ``fmt_minibatch_trajectories`` (reflect.py:121-248).

    ``results`` are SkillOpt rollout results as the mode's package renders
    them (``env.SkillOptEnv.rollout_result``). SkillOpt reads each item's
    ``conversation.json`` and ``target_*_prompt.txt`` from the prediction
    directory; here they travel in the result as ``conversation``,
    ``target_system_prompt`` and ``target_user_prompt``. A result without a
    conversation is skipped but keeps its index, as reflect.py:148-156 skips a
    missing or empty ``conversation.json`` inside ``enumerate``.

    Not ported: the ``codex_trace_summary`` and ``claude_trace_steps`` blocks
    (reflect.py:201-212, :226-232), which SkillOpt adds only from files the
    codex/claude-code exec target backends write, behind environment switches.
    """
    parts: list[str] = []
    for idx, item in enumerate(results, 1):
        tid = str(item["id"])
        conversation = item.get("conversation")
        if not conversation:
            continue

        traj_text = fmt_trajectory(conversation)
        header = (
            f"### Trajectory {idx} (id={tid})\n"
            f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
            f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
        )
        fail_reason = item.get("fail_reason", "")
        if fail_reason:
            header += f"Failure reason: {fail_reason}\n"
        header += f"Steps: {item.get('n_turns', '?')}\n"

        reference_text = str(item.get("reference_text") or "").strip()
        if reference_text:
            header += (
                f"\n#### Hidden Reference\n"
                f"{reference_text}\n"
            )

        target_prompt = item.get("target_system_prompt", "")
        if target_prompt:
            header += (
                f"\n#### Target System Prompt\n"
                f"{target_prompt}\n"
            )

        user_prompt = item.get("target_user_prompt", "")
        if user_prompt:
            header += (
                f"\n#### Target User Prompt\n"
                f"{user_prompt}\n"
            )

        codex_probe_trace_steps = str(item.get("codex_probe_trace_steps") or "").strip()
        if codex_probe_trace_steps:
            header += (
                f"\n#### Codex Trace Steps\n"
                f"{codex_probe_trace_steps}\n"
            )

        preview = item.get("spreadsheet_preview", "")
        if preview:
            header += (
                f"\n#### Spreadsheet Preview\n"
                f"{preview}\n"
            )

        parts.append(header + "\n" + traj_text)

    return "\n\n---\n\n".join(parts)


def format_analyst_user(
    skill_text: str,
    results: list[dict],
    edit_budget: int,
    meta_skill: str,
    *,
    failure: bool,
) -> str:
    """Analyst user message — reflect.py:321-344 (failure) and :421-441 (success).

    ``results`` are the group's rollout results (``fmt_minibatch_trajectories``).
    The "Previous Steps in This Epoch" block (reflect.py:336-340) is empty here:
    this strategy takes one step per epoch, so the step buffer is always empty
    when the analysts run (trainer.py:1188, :1277). Returns ``""`` when no
    result has a trajectory: SkillOpt then makes no call (reflect.py:324-325).
    """
    trajectories_text = fmt_minibatch_trajectories(results)
    if not trajectories_text.strip():
        return ""
    user = f"## Current Skill\n{skill_text}\n\n"
    user += f"## Edits Budget\nProduce at most L={edit_budget} edits.\n\n"
    optimizer_ctx = format_meta_skill_context(meta_skill)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    heading = "Failed Trajectories" if failure else "Successful Trajectories"
    user += f"## {heading} ({len(results)} total)\n{trajectories_text}"
    return user


def format_merge_user(
    skill_text: str,
    patches: list[dict],
    meta_skill: str,
    level: int,
) -> str:
    """Batch-merge user message — aggregate.py:37-44 ``_merge_batch``."""
    patches_text = json.dumps(patches, ensure_ascii=False, indent=2)
    user = (
        f"## Current Skill\n{skill_text}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n{patches_text}"
    )
    optimizer_ctx = format_meta_skill_context(meta_skill)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    return user


def format_merge_final_user(
    skill_text: str,
    failure_merged: dict,
    success_merged: dict,
    n_failure_edits: int,
    n_success_edits: int,
    meta_skill: str,
) -> str:
    """Final failure-first merge user message — aggregate.py:217-242 (patch mode)."""
    combined_text = json.dumps([failure_merged, success_merged], ensure_ascii=False, indent=2)
    user = (
        f"## Current Skill\n{skill_text}\n\n"
        f"## Two pre-merged patch groups to combine\n"
        f"Group 1 (failure-driven, HIGH priority): "
        f"{n_failure_edits} edits\n"
        f"Group 2 (success-driven, lower priority): "
        f"{n_success_edits} edits\n\n"
        f"{combined_text}"
    )
    optimizer_ctx = format_meta_skill_context(meta_skill)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    return user


def describe_edit(edit: dict) -> str:
    """One edit as the ranker sees it — update_modes.py:100-111 (patch mode)."""
    if not isinstance(edit, dict):
        return ""
    op = edit.get("op", "?")
    target = edit.get("target", "")
    content = edit.get("content", "")
    parts = [f"op={op}"]
    if target:
        parts.append(f"target={target!r}")
    if content:
        parts.append(f"content={content!r}")
    if edit.get("support_count") is not None:
        parts.append(f"support={edit.get('support_count')}")
    return "  ".join(parts)


def format_rank_user(
    skill_text: str,
    edits: list[dict],
    budget: int,
    meta_skill: str = "",
) -> str:
    """Rank user message — clip.py:58-71."""
    edits_desc = [f"[{i}] {describe_edit(edit)}" for i, edit in enumerate(edits)]
    user = (
        f"## Current Skill\n{skill_text}\n\n"
        f"## Edits Pool ({len(edits)} edits, budget={budget})\n"
        + "\n".join(edits_desc)
        + f"\n\nSelect the {budget} most important edits. "
        f"Return their 0-based indices in priority order."
    )
    optimizer_ctx = format_meta_skill_context(meta_skill)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"
    return user


def format_comparison_text(pairs: list[dict]) -> str:
    """Longitudinal comparison text — slow_update.py:287-349."""
    by_cat: dict[str, list[dict]] = {
        "regressed": [],
        "persistent_fail": [],
        "improved": [],
        "stable_success": [],
    }
    for p in pairs:
        by_cat.setdefault(p["category"], []).append(p)

    total = len(pairs)
    parts = [
        f"## Longitudinal Comparison Summary\n"
        f"Total samples: {total}\n"
        f"- Improved (wrong→right): {len(by_cat['improved'])}\n"
        f"- Regressed (right→wrong): {len(by_cat['regressed'])}\n"
        f"- Persistent failures (wrong→wrong): {len(by_cat['persistent_fail'])}\n"
        f"- Stable successes (right→right): {len(by_cat['stable_success'])}\n"
    ]

    categories = [
        ("regressed", "Regressions (right→wrong) — HIGHEST PRIORITY", True),
        ("persistent_fail", "Persistent Failures (wrong→wrong)", True),
        ("improved", "Improvements (wrong→right)", True),
        ("stable_success", "Stable Successes (right→right)", False),
    ]

    for cat_key, label, show_traj in categories:
        entries = by_cat[cat_key]
        if not entries:
            parts.append(f"### {label}\n(none)\n")
            continue

        lines = [f"### {label}"]
        for e in entries:
            prev = e["prev"]
            curr = e["curr"]
            lines.append(
                f"\n#### Task {e['id']}: {e['task']}\n"
                f"- Prev epoch: {'PASS' if prev['hard'] else 'FAIL'} "
                f"(soft={prev['soft']:.2f}) — answer: {str(prev['predicted_answer'])}\n"
                f"- Curr epoch: {'PASS' if curr['hard'] else 'FAIL'} "
                f"(soft={curr['soft']:.2f}) — answer: {str(curr['predicted_answer'])}"
            )
            if curr.get("fail_reason"):
                lines.append(f"- Curr fail reason: {curr['fail_reason']}")
            if prev.get("fail_reason") and not prev["hard"]:
                lines.append(f"- Prev fail reason: {prev['fail_reason']}")

            if show_traj:
                if e.get("prev_trajectory"):
                    lines.append(
                        f"\n**Previous epoch trajectory:**\n```\n{e['prev_trajectory']}\n```"
                    )
                if e.get("curr_trajectory"):
                    lines.append(
                        f"\n**Current epoch trajectory:**\n```\n{e['curr_trajectory']}\n```"
                    )

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def format_slow_update_user(
    prev_skill: str,
    current_skill: str,
    prev_guidance: str,
    comparison_text: str,
) -> str:
    """Slow-update user message — slow_update.py:409-424."""
    prev_guidance_section = (
        prev_guidance.strip()
        if prev_guidance and prev_guidance.strip()
        else "(No previous guidance — this is the first slow update.)"
    )
    return (
        f"## Previous Epoch's Skill\n{prev_skill}\n\n"
        f"## Current Epoch's Skill\n{current_skill}\n\n"
        f"## Previous Slow Update Guidance\n"
        f"The following guidance was active during the current epoch. "
        f"Reflect on its effectiveness before writing the new version.\n\n"
        f"{prev_guidance_section}\n\n"
        f"## Longitudinal Comparison (same 20 tasks, two skill versions)\n"
        f"{comparison_text}"
    )


def format_meta_skill_user(
    prev_skill: str,
    current_skill: str,
    prev_meta: str,
    comparison_text: str,
) -> str:
    """Meta-skill user message — meta_skill.py:44-60."""
    prev_meta_section = (
        prev_meta.strip()
        if prev_meta and prev_meta.strip()
        else "(No previous optimizer meta skill — this is the first update.)"
    )
    return (
        f"## Previous Epoch Last-Step Skill\n{prev_skill}\n\n"
        f"## Current Epoch Last-Step Skill\n{current_skill}\n\n"
        f"## Previous Optimizer Meta Skill\n"
        f"The following optimizer memory was available during the current epoch. "
        f"Reflect on whether it improved or harmed the quality of edits.\n\n"
        f"{prev_meta_section}\n\n"
        f"## Longitudinal Comparison (same tasks, two last-step skills)\n"
        f"{comparison_text}"
    )


# ── Slow-update field (slow_update.py:36-93) ───────────────────────────────

def has_slow_update_field(skill: str) -> bool:
    """slow_update.py:42-43."""
    return SLOW_UPDATE_START in skill and SLOW_UPDATE_END in skill


def inject_empty_slow_update_field(skill: str) -> str:
    """Epoch-1 placeholder — slow_update.py:46-53."""
    if has_slow_update_field(skill):
        return skill
    return skill.rstrip() + f"\n\n{SLOW_UPDATE_START}\n{SLOW_UPDATE_END}\n"


def extract_slow_update_field(skill: str) -> str:
    """slow_update.py:56-62."""
    start = skill.find(SLOW_UPDATE_START)
    end = skill.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return ""
    return skill[start + len(SLOW_UPDATE_START):end].strip()


def _strip_all_slow_update_fields(skill: str) -> str:
    """slow_update.py:65-82."""
    while True:
        start = skill.find(SLOW_UPDATE_START)
        if start == -1:
            break
        end = skill.find(SLOW_UPDATE_END, start)
        if end == -1:
            skill = skill[:start] + skill[start + len(SLOW_UPDATE_START):]
            break
        skill = skill[:start] + skill[end + len(SLOW_UPDATE_END):]
    skill = skill.replace(SLOW_UPDATE_END, "")
    while "\n\n\n" in skill:
        skill = skill.replace("\n\n\n", "\n\n")
    return skill.rstrip()


def replace_slow_update_field(skill: str, new_content: str) -> str:
    """Overwrite the protected guidance — slow_update.py:85-93."""
    skill = _strip_all_slow_update_fields(skill)
    return skill + f"\n\n{SLOW_UPDATE_START}\n{new_content.strip()}\n{SLOW_UPDATE_END}\n"
