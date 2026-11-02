"""SkillOpt training loop — the 6-stage ReflACT pipeline ported from SkillOpt (MIT).

Ports the core training loop from ``skillopt/engine/trainer.py`` (MIT):
  1. Rollout   — run solver on training items (uses livemath graph code)
  2. Reflect   — analyze trajectories, generate patches (analyst model calls)
  3. Aggregate — merge patches hierarchically (merge model calls)
  4. Select    — rank and select top edits (ranking model call)
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate on val set, accept/reject gate
  + Slow Update / Meta Skill at epoch end

The ``model`` argument is a callable matching ``complete(prompt, *, system, **kw) -> str``.
In replay mode it is a :class:`ReplayClient`; in a real run it would be a live model.

Source references are to ``~/skillopt`` (MIT) commit at skillopt_commit.txt.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import textwrap
from pathlib import Path
from typing import Any

from refactory_packages.livemath.grading import evaluate as grade_answer
from refactory_packages.livemath.mode import SYSTEM_PROMPT as _SOLVER_SYSTEM_TEMPLATE
from refactory_packages.livemath.ops import _format_choices
from refactory_packages.livemath.shuffling import shuffle_item_choices


# ── prompt templates ported from ~/skillopt/skillopt/prompts/*.md and
#    ~/skillopt/skillopt/envs/livemathematicianbench/prompts/*.md (MIT) ──

# LiveMath-specific analyst prompts (env override)
_ANALYST_ERROR_SYSTEM = textwrap.dedent("""\
You are an expert failure-analysis agent for theorem-grounded mathematical multiple-choice questions.

You will be given MULTIPLE failed trajectories from a single minibatch and the current skill document.
Each trajectory includes the target's response and an evaluation result showing the predicted option
versus the correct option.

Your job is to identify COMMON reasoning failures across the batch and propose concise skill edits.

## Failure Type Categories
- **quantifier_miss**: the agent missed exact quantifiers, scope, or existence/uniqueness conditions
- **strength_mismatch**: the agent preferred a weaker or stronger statement than what was proved
- **condition_miss**: the agent ignored hypotheses, equality cases, or domain restrictions
- **option_confusion**: the agent confused similar answer choices or failed to compare them exactly
- **other**: none of the above

## Rules
1. Focus on patterns that recur across the minibatch.
2. Prefer edits that improve exact choice discrimination, not theorem-specific memorization.
3. Do not hardcode paper-specific content.
4. Only patch gaps not already covered by the skill.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
  "patch": {
    "reasoning": "<why these edits address the common failures>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}""")

_ANALYST_SUCCESS_SYSTEM = textwrap.dedent("""\
You are an expert success-pattern analyst for theorem-grounded mathematical multiple-choice questions.

You will be given MULTIPLE successful trajectories from a minibatch and the current skill document.
Identify generalizable behavior patterns that are genuinely helping the agent choose the exact correct option.

## Rules
- Focus on broadly useful reasoning behaviors.
- Prefer patterns about exact comparison of options, quantifiers, and equality conditions.
- Do not add theorem-specific facts.
- "edits" may be empty if the skill already captures the useful patterns.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number>,
  "success_patterns": ["<pattern 1>", "<pattern 2>"],
  "patch": {
    "reasoning": "<why these patterns matter>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}""")

# Generic merge prompts (from skillopt/prompts/)
_MERGE_FAILURE_SYSTEM = textwrap.dedent("""\
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
}""")

_MERGE_SUCCESS_SYSTEM = textwrap.dedent("""\
You are a skill-edit coordinator. You receive multiple independently-proposed patches
from SUCCESS analysis of agent trajectories. Merge them into ONE coherent, non-redundant patch.

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
      "source_type": "success"
    }
  ]
}""")

_MERGE_FINAL_SYSTEM = textwrap.dedent("""\
You are a skill-edit coordinator performing the FINAL merge. You receive two
pre-merged patch groups (one from FAILURE analysis and one from SUCCESS analysis)
and must combine them into a SINGLE coherent patch.

## Priority
FAILURE-driven edits take priority over SUCCESS-driven edits when they conflict.

Merge guidelines:
1. **Deduplicate** across groups.
2. **Resolve conflicts**: FAILURE edits win.
3. **Preserve unique** SUCCESS insights that don't conflict.
4. **Support count**: propagate from source patches.
5. **PROTECTED SECTION**: Do NOT edit content between <!-- SLOW_UPDATE_START -->
   and <!-- SLOW_UPDATE_END -->.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<consolidation summary>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "failure|success"
    }
  ]
}""")

_RANKING_SYSTEM = textwrap.dedent("""\
You are an expert skill-optimization optimizer. You receive a skill document and a pool
of proposed edits. Your job is to RANK the edits by importance and select the top ones.

Ranking criteria (in order of priority):
1. **Systematic impact**: edits that address widespread, recurring failure patterns
   across many tasks should rank highest. A rule that fixes 50% of failures beats
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
}""")

_SLOW_UPDATE_SYSTEM = textwrap.dedent("""\
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
   - Retains effective parts from the previous guidance.
   - Drops or corrects parts that failed.
   - Adds new insights from the current longitudinal comparison.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<reflection on previous guidance + analysis of longitudinal data>",
  "slow_update_content": "<concise strategic guidance for next epoch>"
}""")

_META_SKILL_SYSTEM = textwrap.dedent("""\
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

Respond ONLY with a valid JSON object:
{
  "reasoning": "<analysis of what worked and what didn't>",
  "meta_skill_content": "<concise optimizer guidance for future epochs>"
}""")


# ── utilities ported from ~/skillopt/skillopt/utils/ (MIT) ────────────────

def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text.

    Ported from skillopt/utils/json_utils.py:extract_json (MIT).
    """
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _skill_hash(content: str) -> str:
    """16-char hex hash matching A's format.

    Ported from skillopt/utils/scoring.py:skill_hash (MIT).
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── solver prompt construction (delegates to livemath) ────────────────────

def build_solver_system(skill_text: str) -> str:
    """Build the solver system prompt using B's livemath template.

    Matches A's rollout.py:16-21 (MIT): ``_build_system(skill_content)``.
    """
    if skill_text.strip():
        skill_section = f"## Skill\n{skill_text.strip()}\n\n"
    else:
        skill_section = ""
    return _SOLVER_SYSTEM_TEMPLATE.replace("{{skill_section}}", skill_section).rstrip()


def build_solver_user(item: dict) -> str:
    """Build the solver user prompt using B's livemath ops.

    Matches A's rollout.py:31-53 (MIT): ``_build_user(item)``.
    """
    question = item.get("question", "")
    choices = item.get("choices", [])
    choices_text = _format_choices(choices)
    return f"## Question\n{question}\n\n## Choices\n{choices_text}"


# ── trajectory formatting ported from ~/skillopt/skillopt/gradient/reflect.py (MIT) ──

def _fmt_trajectory(conversation: list[dict]) -> str:
    """Format a single conversation into analyst-readable text.

    Ported from skillopt/gradient/reflect.py:fmt_trajectory (MIT).
    """
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {item}")
            continue
        if item.get("type") == "tool_call":
            lines.append(f"[action] {item.get('cmd', '')}")
            lines.append(f"[obs]    {item.get('obs', '')}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = item.get("reasoning", "")
            action = item.get("action", "")
            feedback = item.get("env_feedback", "")
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif item.get("role") == "system":
            lines.append(f"[verification] {item.get('content', '')}")
        else:
            role = item.get("role", "agent")
            lines.append(f"[{role}] {item.get('content', '')}")
    return "\n".join(lines)


def _fmt_minibatch_trajectories(
    results: list[dict],
    conversations: dict[str, list[dict]],
    system_prompts: dict[str, str] | None = None,
    user_prompts: dict[str, str] | None = None,
) -> str:
    """Format multiple trajectories for minibatch analyst consumption.

    Ported from skillopt/gradient/reflect.py:fmt_minibatch_trajectories (MIT).
    Matches A's exact format:
      ``### Trajectory {idx} (id={tid})``
      ``Task: {task_description}``
      ``Task type: {task_type}``
      ``Failure reason: {fail_reason}``
      ``Steps: {n_turns}``
      ``#### Target System Prompt``
      ``#### Target User Prompt``
      then the conversation text.

    Trajectories separated by ``\\n\\n---\\n\\n``.
    """
    system_prompts = system_prompts or {}
    user_prompts = user_prompts or {}
    parts: list[str] = []

    for idx, item in enumerate(results, 1):
        tid = str(item["id"])

        header = (
            f"### Trajectory {idx} (id={tid})\n"
            f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
            f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
        )
        fail_reason = item.get("fail_reason", "")
        if fail_reason:
            header += f"Failure reason: {fail_reason}\n"
        header += f"Steps: {item.get('n_turns', '?')}\n"

        # Target system prompt (what the solver saw).
        # A writes the system prompt to target_system_prompt.txt with
        # a trailing newline (from the template file), then reads it
        # back with f.read(). We match this by ensuring the embedded
        # system prompt ends with \n before the separator.
        # Ported from skillopt/gradient/reflect.py:183-187 (MIT).
        sys_prompt = system_prompts.get(tid, "")
        if sys_prompt:
            # Ensure trailing newline to match A's file-written system prompt
            if not sys_prompt.endswith("\n"):
                sys_prompt += "\n"
            header += f"\n#### Target System Prompt\n{sys_prompt}\n"

        # Target user prompt (what the solver saw)
        usr_prompt = user_prompts.get(tid, "")
        if usr_prompt:
            header += f"\n#### Target User Prompt\n{usr_prompt}\n"

        conv = conversations.get(tid, [])
        if conv:
            traj_text = _fmt_trajectory(conv)
            parts.append(header + "\n" + traj_text)
        else:
            resp = item.get("response", "")
            if resp:
                parts.append(header + "\n" + f"[agent] {resp}")
            else:
                parts.append(header)

    return "\n\n---\n\n".join(parts)


# ── edit application ported from ~/skillopt/skillopt/optimizer/skill.py (MIT) ──

def _apply_edit(skill: str, edit: dict) -> tuple[str, str]:
    """Apply a single edit to the skill. Returns (new_skill, status).

    Ported from skillopt/optimizer/skill.py:apply_patch_with_report (MIT).
    """
    op = edit.get("op", "")
    content = edit.get("content", "")
    target = edit.get("target", "")

    if op == "append":
        return skill.rstrip() + "\n\n" + content + "\n", "applied"
    elif op == "insert_after":
        if target and target in skill:
            idx = skill.index(target) + len(target)
            return skill[:idx] + "\n" + content + skill[idx:], "applied"
        return skill, "skipped_target_not_found"
    elif op == "replace":
        if target and target in skill:
            return skill.replace(target, content, 1), "applied"
        return skill, "skipped_target_not_found"
    elif op == "delete":
        if target and target in skill:
            return skill.replace(target, "", 1), "applied"
        return skill, "skipped_target_not_found"
    return skill, f"skipped_unknown_op_{op}"


def apply_edits(skill: str, edits: list[dict]) -> tuple[str, list[dict]]:
    """Apply a list of edits to the skill. Returns (new_skill, apply_report).

    Ported from skillopt/optimizer/skill.py:apply_patch_with_report (MIT).
    """
    report: list[dict] = []
    for edit in edits:
        skill, status = _apply_edit(skill, edit)
        report.append({
            "op": edit.get("op", ""),
            "target": edit.get("target", "")[:80],
            "content": edit.get("content", "")[:80],
            "status": status,
        })
    return skill, report


# ── dataloader helpers ported from ~/skillopt/skillopt/envs/livemathematicianbench/dataloader.py (MIT) ──

def shuffle_epoch_items(
    items: list[dict], seed: int, epoch: int,
) -> list[dict]:
    """Shuffle items for an epoch, matching A's dataloader.

    Ported from skillopt/envs/livemathematicianbench/dataloader.py:245-247 (MIT).
    """
    epoch_rng = random.Random(seed + epoch * 1000)
    shuffled = list(items)
    epoch_rng.shuffle(shuffled)
    return shuffled


def compute_batch_seed(seed: int, epoch: int, batch_idx: int = 0) -> int:
    """Compute the per-batch choice-shuffle seed matching A's dataloader.

    Ported from skillopt/envs/livemathematicianbench/dataloader.py:255 (MIT).
    """
    return seed + epoch * 1000 + batch_idx + 1


# ── meta-skill context formatting ──

def _format_meta_skill_context(meta_skill: str) -> str:
    """Format meta-skill content for injection into optimizer prompts.

    Ported from skillopt/optimizer/meta_skill.py:format_meta_skill_context (MIT).
    """
    if not meta_skill or not meta_skill.strip():
        return ""
    return (
        "## Optimizer Meta Skill (from previous epoch)\n"
        "Use the following guidance to improve your analysis and edit proposals:\n\n"
        f"{meta_skill.strip()}"
    )


# ── edit description for ranking ──

def _describe_edit(edit: dict) -> str:
    """Describe an edit for the ranking prompt.

    Ported from skillopt/optimizer/clip.py:describe_item (MIT).
    """
    op = edit.get("op", "?")
    target = edit.get("target", "")
    content = edit.get("content", "")
    support = edit.get("support_count", "?")
    source = edit.get("source_type", "?")
    if op == "append":
        return f'[{op}] ({source}, support={support}) "{content[:100]}"'
    elif op in ("insert_after", "replace"):
        return f'[{op}] ({source}, support={support}) target="{target[:60]}" → "{content[:80]}"'
    elif op == "delete":
        return f'[{op}] ({source}, support={support}) target="{target[:100]}"'
    return f'[{op}] ({source}, support={support}) "{content[:100]}"'


# ── The training loop ────────────────────────────────────────────────────

class SkillOptTrainer:
    """SkillOpt 6-stage training loop.

    Ported from ``~/skillopt/skillopt/engine/trainer.py`` (MIT).

    Parameters
    ----------
    model : object
        Model client with ``complete(prompt, *, system, **kw) -> str``.
    config : dict
        Training config mapped from A's config.json via ``load_a_config()``.
    train_items : list[dict]
        Training items from the split.
    val_items : list[dict]
        Validation items from the split.
    initial_skill : str
        Initial skill text.
    output_dir : Path
        Directory for B's output files.
    """

    def __init__(
        self,
        model: Any,
        config: dict[str, Any],
        train_items: list[dict[str, Any]],
        val_items: list[dict[str, Any]],
        initial_skill: str,
        output_dir: Path,
        *,
        a_out_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_items = train_items
        self.val_items = val_items
        self.initial_skill = initial_skill
        self.output_dir = Path(output_dir)
        # A's out/ directory — used to read results ordering (A ran concurrently)
        self.a_out_dir = Path(a_out_dir) if a_out_dir else None

        # State
        self.current_skill = initial_skill
        self.best_skill = initial_skill
        self.best_score: float = -1.0
        self.current_score: float = -1.0
        self.best_step: int = 0

        # Config
        self.seed = config.get("seed", 42)
        self.max_epochs = config.get("max_epochs", 10)
        self.early_stop_patience = config.get("early_stop_patience", 0)
        self.early_stop_at_full_val = config.get("early_stop_at_full_val", False)
        self.edit_budget_max = config.get("edit_budget_max", 4)
        self.edit_budget_min = config.get("edit_budget_min", 2)
        self.group_size = config.get("group_size", 8)
        self.merge_batch_size = config.get("merge_batch_size", 8)
        self.slow_update_samples = config.get("slow_update_samples", 20)
        self.use_slow_update = config.get("use_slow_update", True)
        self.use_meta_skill = config.get("use_meta_skill", True)

        # LR scheduler: cosine from edit_budget_max to edit_budget_min
        # Ported from skillopt/optimizer/scheduler.py:build_scheduler (MIT)
        self.total_steps = self.max_epochs  # 1 step per epoch
        self._meta_skill_content = ""
        self._slow_update_content = ""

    def _lr_step(self, step: int) -> int:
        """Cosine LR schedule.

        Ported from skillopt/optimizer/scheduler.py:CosineScheduler (MIT).
        """
        import math
        if self.total_steps <= 1:
            return self.edit_budget_max
        progress = step / self.total_steps
        cos_val = (1 + math.cos(math.pi * progress)) / 2
        lr = self.edit_budget_min + (self.edit_budget_max - self.edit_budget_min) * cos_val
        return max(self.edit_budget_min, round(lr))

    def run(self) -> list[dict[str, Any]]:
        """Run the full training loop. Returns B's history entries."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Baseline evaluation
        if self.current_score < 0 and self.val_items:
            self.current_score = self._evaluate_skill(self.initial_skill, "baseline")
            self.best_score = self.current_score

        history: list[dict[str, Any]] = []
        es_stale = 0
        es_best = self.best_score

        for epoch in range(1, self.max_epochs + 1):
            entry = self._run_epoch(epoch, history)
            history.append(entry)

            # Early stop check
            # Ported from trainer.py:1168-1174 (MIT)
            if self.early_stop_patience > 0:
                if self.best_score > es_best:
                    es_best = self.best_score
                    es_stale = 0
                else:
                    es_stale += 1
                if es_stale >= self.early_stop_patience:
                    entry["early_stop"] = True
                    break

        return history

    def _run_epoch(self, epoch: int, history: list[dict]) -> dict[str, Any]:
        """Run one epoch of the training loop."""
        step = epoch  # A uses step = epoch (1-indexed)
        step_dir = self.output_dir / "steps" / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        entry: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
        }

        # Load A's result ordering for this step (A ran concurrently).
        # This ensures B's failure/success grouping matches A's.
        a_result_order = self._load_a_result_order(step)

        # ── 1. ROLLOUT ────────────────────────────────────────────────
        epoch_items = shuffle_epoch_items(self.train_items, self.seed, epoch)
        batch_seed = compute_batch_seed(self.seed, epoch)
        rollout_results, conversations, sys_prompts, usr_prompts = self._run_rollout(
            epoch_items, self.current_skill, batch_seed, step_dir / "rollout",
        )
        n_items = len(rollout_results)
        hard_score = (
            sum(float(r.get("hard", 0)) for r in rollout_results) / n_items
            if n_items > 0 else 0.0
        )
        entry["rollout_hard"] = round(hard_score, 6)
        entry["rollout_n"] = n_items

        # Reorder results to match A's concurrent completion order.
        # A ran solver calls concurrently; results.jsonl records completion
        # order. B runs sequentially. Without this reorder, the analyst
        # minibatch grouping would differ, causing prompt mismatches.
        if a_result_order:
            id_to_result = {str(r["id"]): r for r in rollout_results}
            reordered = [id_to_result[aid] for aid in a_result_order if aid in id_to_result]
            # Append any items not in A's order (shouldn't happen normally)
            seen = set(a_result_order)
            for r in rollout_results:
                if str(r["id"]) not in seen:
                    reordered.append(r)
            rollout_results = reordered

        # Split into failures and successes
        failures = [r for r in rollout_results if not r.get("hard")]
        successes = [r for r in rollout_results if r.get("hard")]

        # ── 2. REFLECT ────────────────────────────────────────────────
        # A shuffles failures and successes before minibatch grouping.
        # Ported from skillopt/gradient/reflect.py:_shuffle_for_minibatch (MIT).
        fail_rng = random.Random(batch_seed)
        shuffled_failures = list(failures)
        fail_rng.shuffle(shuffled_failures)

        succ_rng = random.Random(batch_seed + 1)
        shuffled_successes = list(successes)
        succ_rng.shuffle(shuffled_successes)

        failure_patches, success_patches, all_raw = self._reflect(
            shuffled_failures, shuffled_successes, conversations, self.current_skill,
            self._lr_step(step), step_dir,
            system_prompts=sys_prompts, user_prompts=usr_prompts,
        )
        n_patches = len(failure_patches) + len(success_patches)
        entry["n_patches"] = n_patches
        entry["n_failure_patches"] = len(failure_patches)
        entry["n_success_patches"] = len(success_patches)

        if n_patches == 0:
            entry["action"] = "skip_no_patches"
            entry["current_score"] = self.current_score
            entry["best_score"] = self.best_score
            entry["best_step"] = self.best_step
            return entry

        # ── 3. AGGREGATE ──────────────────────────────────────────────
        merged_patch = self._aggregate(
            failure_patches, success_patches, self.current_skill, step_dir,
        )
        merged_edits = merged_patch.get("edits", [])
        entry["n_edits_merged"] = len(merged_edits)

        # ── 4. SELECT ────────────────────────────────────────────────
        edit_budget = self._lr_step(step)
        ranked_edits = self._select(
            self.current_skill, merged_edits, edit_budget, step_dir,
        )
        entry["n_edits_ranked"] = len(ranked_edits)
        entry["edit_budget"] = edit_budget

        # ── 5. UPDATE ────────────────────────────────────────────────
        candidate_skill, apply_report = apply_edits(self.current_skill, ranked_edits)
        cand_hash = _skill_hash(candidate_skill)
        entry["candidate_hash"] = cand_hash
        entry["candidate_skill_len"] = len(candidate_skill)
        if apply_report:
            entry["edit_apply_summary"] = {
                "total": len(apply_report),
                "applied": sum(1 for r in apply_report if r["status"].startswith("applied")),
                "skipped": sum(1 for r in apply_report if r["status"].startswith("skipped")),
                "errors": sum(1 for r in apply_report if r["status"] == "error"),
            }

        # Save candidate skill
        (step_dir / "candidate_skill.md").write_text(candidate_skill)
        (step_dir / "merged_patch.json").write_text(json.dumps(merged_patch, indent=2))
        (step_dir / "ranked_edits.json").write_text(
            json.dumps({"edits": ranked_edits}, indent=2)
        )
        if apply_report:
            (step_dir / "edit_apply_report.json").write_text(
                json.dumps(apply_report, indent=2)
            )

        # ── 6. EVALUATE ──────────────────────────────────────────────
        if self.val_items:
            cand_hard = self._evaluate_skill(
                candidate_skill, f"step_{step:04d}",
            )
        else:
            cand_hard = hard_score  # fallback

        entry["selection_hard"] = cand_hard

        # Accept/reject gate
        # Ported from skillopt/evaluation/gate.py (MIT)
        if cand_hard > self.best_score:
            entry["action"] = "accept_new_best"
            self.current_skill = candidate_skill
            self.current_score = cand_hard
            self.best_skill = candidate_skill
            self.best_score = cand_hard
            self.best_step = step
        elif cand_hard > self.current_score:
            entry["action"] = "accept"
            self.current_skill = candidate_skill
            self.current_score = cand_hard
        else:
            entry["action"] = "reject"

        entry["current_score"] = self.current_score
        entry["best_score"] = self.best_score
        entry["best_step"] = self.best_step

        # Save skill
        skills_dir = self.output_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / f"skill_v{step:04d}.md").write_text(
            self.current_skill
        )

        # ── SLOW UPDATE (epoch end) ──────────────────────────────────
        if self.use_slow_update and epoch >= 2:
            slow_result = self._slow_update(epoch, history)
            if slow_result and slow_result.get("slow_update_content"):
                self._slow_update_content = slow_result["slow_update_content"]
                entry["slow_update"] = True

        # ── META SKILL (epoch end) ────────────────────────────────────
        if self.use_meta_skill and epoch >= 2:
            meta_result = self._meta_skill(epoch, history)
            if meta_result and meta_result.get("meta_skill_content"):
                self._meta_skill_content = meta_result["meta_skill_content"]
                entry["meta_skill"] = True
                # Save meta skill result
                meta_dir = self.output_dir / "meta_skill" / f"epoch_{epoch:02d}"
                meta_dir.mkdir(parents=True, exist_ok=True)
                (meta_dir / "meta_skill_result.json").write_text(
                    json.dumps(meta_result, indent=2)
                )

        return entry

    def _load_a_result_order(self, step: int) -> list[str] | None:
        """Load A's rollout result IDs in concurrent completion order.

        A's solver calls ran concurrently, so results.jsonl records completion
        order (not input order). We reorder B's results to match, so the
        analyst minibatch grouping is the same.
        """
        if self.a_out_dir is None:
            return None
        results_path = self.a_out_dir / "steps" / f"step_{step:04d}" / "rollout" / "results.jsonl"
        if not results_path.exists():
            return None
        ids: list[str] = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)["id"])
        return ids

    def _run_rollout(
        self,
        items: list[dict],
        skill_text: str,
        batch_seed: int,
        rollout_dir: Path,
    ) -> tuple[list[dict], dict[str, list[dict]], dict[str, str], dict[str, str]]:
        """Run the solver on items via the model client.

        Uses B's livemath prompt construction (mode.py + ops.py).
        Returns (results, conversations_by_id, system_prompts_by_id, user_prompts_by_id).
        """
        rollout_dir.mkdir(parents=True, exist_ok=True)
        system = build_solver_system(skill_text)
        results: list[dict[str, Any]] = []
        conversations: dict[str, list[dict]] = {}
        system_prompts: dict[str, str] = {}
        user_prompts: dict[str, str] = {}

        for item in items:
            shuffled = shuffle_item_choices(item, batch_seed)
            prompt = build_solver_user(shuffled)
            response = self.model.complete(prompt, system=system)

            # Grade
            grade = grade_answer(
                response,
                shuffled["correct_choice"],
                shuffled["choices"],
            )

            tid = str(item["id"])

            # Build conversation matching A's format
            # Ported from rollout.py:246-284 (MIT)
            conversation: list[dict] = [
                {"type": "message", "turn": 1, "content": response},
            ]
            eval_detail = (
                f"[EVALUATION RESULT]\n"
                f"Question: {item['question']}\n"
                f"Predicted label: {grade['predicted_label']!r}\n"
                f"Predicted text: {grade.get('predicted_text', '')!r}\n"
                f"Correct label: {grade['correct_label']!r}\n"
                f"Correct text: {grade.get('correct_text', '')!r}\n"
                f"Exact Match: {grade['em']}"
            )
            conversation.append({"role": "system", "content": eval_detail})
            conversations[tid] = conversation
            system_prompts[tid] = system
            user_prompts[tid] = prompt

            fail_reason = ""
            if not grade["em"]:
                fail_reason = (
                    f"MCQ=0: predicted '{grade.get('predicted_label') or grade.get('predicted_answer', '')}' "
                    f"but expected '{grade['correct_label']}'"
                )

            # task_type ported from rollout.py:132 (MIT):
            # item.get("theorem_type", ["math_mcq"])[0]
            tt = item.get("theorem_type", ["math_mcq"])
            task_type = tt[0] if isinstance(tt, list) and tt else "math_mcq"

            results.append({
                "id": tid,
                "question": item["question"],
                "task_type": task_type,
                "hard": int(grade["em"]),
                "soft": grade.get("f1", grade["em"]),
                "predicted_answer": grade.get("predicted_answer", ""),
                "predicted_label": grade.get("predicted_label", ""),
                "predicted_text": grade.get("predicted_text", ""),
                "correct_label": grade["correct_label"],
                "correct_text": grade.get("correct_text", ""),
                "response": response[:2000],
                "fail_reason": fail_reason,
                "agent_ok": True,
                "n_turns": 1,
            })

        # Save results
        with open(rollout_dir / "results.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results, conversations, system_prompts, user_prompts

    def _reflect(
        self,
        failures: list[dict],
        successes: list[dict],
        conversations: dict[str, list[dict]],
        skill_text: str,
        edit_budget: int,
        step_dir: Path,
        *,
        system_prompts: dict[str, str] | None = None,
        user_prompts: dict[str, str] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict | None]]:
        """Reflect: analyse failures and successes in minibatches.

        Ported from skillopt/gradient/reflect.py (MIT).
        Returns (failure_patches, success_patches, all_raw).
        """
        all_raw: list[dict | None] = []
        failure_patches: list[dict] = []
        success_patches: list[dict] = []

        # Group failures into minibatches
        for i in range(0, len(failures), self.group_size):
            batch = failures[i:i + self.group_size]
            traj_text = _fmt_minibatch_trajectories(
                batch, conversations, system_prompts, user_prompts,
            )
            if not traj_text.strip():
                continue

            user = (
                f"## Current Skill\n{skill_text}\n\n"
                f"## Edits Budget\n"
                f"Produce at most L={edit_budget} edits.\n\n"
            )
            meta_ctx = _format_meta_skill_context(self._meta_skill_content)
            if meta_ctx:
                user += meta_ctx + "\n\n"
            user += f"## Failed Trajectories ({len(batch)} total)\n{traj_text}"

            response = self.model.complete(user, system=_ANALYST_ERROR_SYSTEM)
            result = _extract_json(response)
            if result and "patch" in result:
                result["source_type"] = "failure"
                patch = result["patch"]
                edits = patch.get("edits", [])
                # Propagate source_type and support_count to individual edits,
                # matching A's _normalise_patches (skillopt/engine/trainer.py:157-186).
                support = max(int(result.get("batch_size", 0) or 0), 1)
                for item in edits:
                    if isinstance(item, dict):
                        item.setdefault("source_type", "failure")
                        item.setdefault("support_count", support)
                if edits:
                    failure_patches.append(patch)
                all_raw.append(result)
            else:
                all_raw.append(None)

        # Group successes into minibatches
        for i in range(0, len(successes), self.group_size):
            batch = successes[i:i + self.group_size]
            traj_text = _fmt_minibatch_trajectories(
                batch, conversations, system_prompts, user_prompts,
            )
            if not traj_text.strip():
                continue

            user = (
                f"## Current Skill\n{skill_text}\n\n"
                f"## Edits Budget\n"
                f"Produce at most L={edit_budget} edits.\n\n"
            )
            meta_ctx = _format_meta_skill_context(self._meta_skill_content)
            if meta_ctx:
                user += meta_ctx + "\n\n"
            user += f"## Successful Trajectories ({len(batch)} total)\n{traj_text}"

            response = self.model.complete(user, system=_ANALYST_SUCCESS_SYSTEM)
            result = _extract_json(response)
            if result and "patch" in result:
                result["source_type"] = "success"
                patch = result["patch"]
                edits = patch.get("edits", [])
                # Propagate source_type and support_count to individual edits,
                # matching A's _normalise_patches (skillopt/engine/trainer.py:157-186).
                support = max(int(result.get("batch_size", 0) or 0), 1)
                for item in edits:
                    if isinstance(item, dict):
                        item.setdefault("source_type", "success")
                        item.setdefault("support_count", support)
                if edits:
                    success_patches.append(patch)
                all_raw.append(result)
            else:
                all_raw.append(None)

        return failure_patches, success_patches, all_raw

    def _aggregate(
        self,
        failure_patches: list[dict],
        success_patches: list[dict],
        skill_text: str,
        step_dir: Path,
    ) -> dict:
        """Aggregate: merge patches hierarchically.

        Ported from skillopt/gradient/aggregate.py (MIT).
        """
        def _merge_group(patches: list[dict], system: str, level: int = 1) -> dict:
            """Merge a group of patches with one LLM call."""
            if len(patches) <= 1:
                return patches[0] if patches else {"edits": []}

            # Hierarchical merge in batches
            while len(patches) > 1:
                new_patches = []
                for i in range(0, len(patches), self.merge_batch_size):
                    batch = patches[i:i + self.merge_batch_size]
                    if len(batch) == 1:
                        new_patches.append(batch[0])
                        continue
                    patches_text = json.dumps(batch, ensure_ascii=False, indent=2)
                    user = (
                        f"## Current Skill\n{skill_text}\n\n"
                        f"## Patches to merge ({len(batch)} total, merge level {level})\n"
                        f"{patches_text}"
                    )
                    meta_ctx = _format_meta_skill_context(self._meta_skill_content)
                    if meta_ctx:
                        user = f"{meta_ctx}\n\n{user}"
                    response = self.model.complete(user, system=system)
                    merged = _extract_json(response)
                    if merged and "edits" in merged:
                        new_patches.append(merged)
                    else:
                        # Fallback: concatenate
                        all_edits = []
                        for p in batch:
                            all_edits.extend(p.get("edits", []))
                        new_patches.append({"edits": all_edits})
                patches = new_patches
                level += 1
            return patches[0]

        # Merge failures
        merged_failure = _merge_group(failure_patches, _MERGE_FAILURE_SYSTEM)
        # Merge successes
        merged_success = _merge_group(success_patches, _MERGE_SUCCESS_SYSTEM)

        # Final merge if both have edits
        f_edits = merged_failure.get("edits", [])
        s_edits = merged_success.get("edits", [])

        if f_edits and s_edits:
            patches_text = json.dumps(
                [merged_failure, merged_success], ensure_ascii=False, indent=2
            )
            user = (
                f"## Current Skill\n{skill_text}\n\n"
                f"## Patches to merge (2 total, merge level final)\n"
                f"{patches_text}"
            )
            meta_ctx = _format_meta_skill_context(self._meta_skill_content)
            if meta_ctx:
                user = f"{meta_ctx}\n\n{user}"
            response = self.model.complete(user, system=_MERGE_FINAL_SYSTEM)
            merged = _extract_json(response)
            if merged and "edits" in merged:
                return merged
            return {"edits": f_edits + s_edits}

        if f_edits:
            return merged_failure
        if s_edits:
            return merged_success
        return {"edits": []}

    def _select(
        self,
        skill_text: str,
        merged_edits: list[dict],
        edit_budget: int,
        step_dir: Path,
    ) -> list[dict]:
        """Select: rank and select top edits.

        Ported from skillopt/optimizer/clip.py:rank_and_select (MIT).
        """
        if len(merged_edits) <= edit_budget:
            return merged_edits

        edits_desc = []
        for i, edit in enumerate(merged_edits):
            edits_desc.append(f"[{i}] {_describe_edit(edit)}")

        user = (
            f"## Current Skill\n{skill_text}\n\n"
            f"## Edit Pool ({len(merged_edits)} edits, budget={edit_budget})\n"
            + "\n".join(edits_desc)
            + f"\n\nSelect the {edit_budget} most important edits. "
            f"Return their 0-based indices in priority order."
        )
        meta_ctx = _format_meta_skill_context(self._meta_skill_content)
        if meta_ctx:
            user = f"{meta_ctx}\n\n{user}"

        response = self.model.complete(user, system=_RANKING_SYSTEM)
        result = _extract_json(response)
        if result and "selected_indices" in result:
            indices = result["selected_indices"]
            selected = []
            seen: set[int] = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(merged_edits) and idx not in seen:
                    selected.append(merged_edits[idx])
                    seen.add(idx)
                if len(selected) >= edit_budget:
                    break
            if selected:
                return selected

        # Fallback: truncate
        return merged_edits[:edit_budget]

    def _evaluate_skill(self, skill_text: str, label: str) -> float:
        """Evaluate a skill on the validation set.

        Runs the solver on val items and returns hard accuracy.
        """
        system = build_solver_system(skill_text)
        correct = 0
        total = len(self.val_items)
        # Use a fixed seed for val evaluation (matches A's behavior)
        val_seed = self.seed

        for item in self.val_items:
            shuffled = shuffle_item_choices(item, val_seed)
            prompt = build_solver_user(shuffled)
            response = self.model.complete(prompt, system=system)
            grade = grade_answer(
                response,
                shuffled["correct_choice"],
                shuffled["choices"],
            )
            if grade["em"]:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _slow_update(self, epoch: int, history: list[dict]) -> dict | None:
        """Run slow update at epoch end.

        Ported from skillopt/engine/trainer.py slow update section (MIT).
        """
        if epoch < 2:
            return None

        # Get previous epoch's skill
        prev_entries = [h for h in history if h.get("epoch") == epoch - 1]
        if not prev_entries:
            return None

        prev_step = prev_entries[-1]["step"]
        prev_skill_path = self.output_dir / "skills" / f"skill_v{prev_step:04d}.md"
        if not prev_skill_path.exists():
            return None
        prev_skill = prev_skill_path.read_text()

        # Sample items for comparison
        slow_seed = self.seed + epoch * 2000
        rng = random.Random(slow_seed)
        sample_items = list(self.train_items)
        rng.shuffle(sample_items)
        sample_items = sample_items[:self.slow_update_samples]

        # Run both skills on sampled items
        prev_system = build_solver_system(prev_skill)
        curr_system = build_solver_system(self.current_skill)

        comparison_parts: list[str] = []
        for item in sample_items:
            shuffled = shuffle_item_choices(item, slow_seed)
            prompt = build_solver_user(shuffled)

            prev_resp = self.model.complete(prompt, system=prev_system)
            curr_resp = self.model.complete(prompt, system=curr_system)

            prev_grade = grade_answer(prev_resp, shuffled["correct_choice"], shuffled["choices"])
            curr_grade = grade_answer(curr_resp, shuffled["correct_choice"], shuffled["choices"])

            prev_correct = bool(prev_grade["em"])
            curr_correct = bool(curr_grade["em"])

            if prev_correct and not curr_correct:
                cat = "regression"
            elif not prev_correct and curr_correct:
                cat = "improvement"
            elif not prev_correct and not curr_correct:
                cat = "persistent_fail"
            else:
                cat = "stable_success"

            comparison_parts.append(
                f"- {item['id']} ({cat}): prev={'correct' if prev_correct else 'wrong'}, "
                f"curr={'correct' if curr_correct else 'wrong'}"
            )

        comparison_text = "\n".join(comparison_parts)

        user = (
            f"## Previous Epoch's Skill\n{prev_skill}\n\n"
            f"## Current Epoch's Skill\n{self.current_skill}\n\n"
            f"## Previous Slow Update Guidance\n{self._slow_update_content or '(none)'}\n\n"
            f"## Longitudinal Comparison ({len(sample_items)} items)\n{comparison_text}"
        )

        response = self.model.complete(user, system=_SLOW_UPDATE_SYSTEM)
        return _extract_json(response)

    def _meta_skill(self, epoch: int, history: list[dict]) -> dict | None:
        """Run meta skill at epoch end.

        Ported from skillopt/engine/trainer.py meta skill section (MIT).
        """
        if epoch < 2:
            return None

        prev_entries = [h for h in history if h.get("epoch") == epoch - 1]
        if not prev_entries:
            return None

        prev_step = prev_entries[-1]["step"]
        prev_skill_path = self.output_dir / "skills" / f"skill_v{prev_step:04d}.md"
        if not prev_skill_path.exists():
            return None
        prev_skill = prev_skill_path.read_text()

        # Use same comparison data as slow update
        meta_seed = self.seed + epoch * 2000
        rng = random.Random(meta_seed)
        sample_items = list(self.train_items)
        rng.shuffle(sample_items)
        sample_items = sample_items[:self.slow_update_samples]

        prev_system = build_solver_system(prev_skill)
        curr_system = build_solver_system(self.current_skill)

        comparison_parts: list[str] = []
        for item in sample_items:
            shuffled = shuffle_item_choices(item, meta_seed)
            prompt = build_solver_user(shuffled)

            prev_resp = self.model.complete(prompt, system=prev_system)
            curr_resp = self.model.complete(prompt, system=curr_system)

            prev_grade = grade_answer(prev_resp, shuffled["correct_choice"], shuffled["choices"])
            curr_grade = grade_answer(curr_resp, shuffled["correct_choice"], shuffled["choices"])

            prev_correct = bool(prev_grade["em"])
            curr_correct = bool(curr_grade["em"])

            if prev_correct and not curr_correct:
                cat = "regression"
            elif not prev_correct and curr_correct:
                cat = "improvement"
            elif not prev_correct and not curr_correct:
                cat = "persistent_fail"
            else:
                cat = "stable_success"

            comparison_parts.append(
                f"- {item['id']} ({cat}): prev={'correct' if prev_correct else 'wrong'}, "
                f"curr={'correct' if curr_correct else 'wrong'}"
            )

        comparison_text = "\n".join(comparison_parts)

        user = (
            f"## Previous Epoch's Skill\n{prev_skill}\n\n"
            f"## Current Epoch's Skill\n{self.current_skill}\n\n"
            f"## Previous Meta Skill\n{self._meta_skill_content or '(none)'}\n\n"
            f"## Longitudinal Comparison ({len(sample_items)} items)\n{comparison_text}"
        )

        response = self.model.complete(user, system=_META_SKILL_SYSTEM)
        return _extract_json(response)
