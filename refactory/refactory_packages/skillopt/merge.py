"""Three-phase failure-first hierarchical merge and LLM rank.

Reproduces ``aggregate.py:195-275`` (hierarchical merge) and ``clip.py:25-109``
(rank and select).
"""

from __future__ import annotations

import json
from typing import Any

from refactory_packages.skillopt import prompts
from refactory_packages.skillopt.trace import TraceRecorder


# ── hierarchical merge ────────────────────────────────────────────────────

def hierarchical_merge(
    client: Any,
    skill_text: str,
    failure_edits: list[dict[str, Any]],
    success_edits: list[dict[str, Any]],
    meta_skill: str,
    epoch: int,
    trace: TraceRecorder,
    merge_batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Failure-first hierarchical merge of analyst edits.

    Phase 1: merge failure patches hierarchically (aggregate.py:195-199).
    Phase 2: merge success patches hierarchically (aggregate.py:201-205).
    Phase 3: final merge with failure priority (aggregate.py:217-275).
    Fallback: concatenation failure-first then truncation (aggregate.py:272-275).
    """
    call_order = [0]

    def _merge_group(edits_list: list[dict], system: str, step_type: str) -> list[dict]:
        """Hierarchically merge until one patch remains."""
        if len(edits_list) <= 1:
            return edits_list
        batches = _chunk(edits_list, merge_batch_size)
        if len(batches) <= 1:
            # Single batch — merge directly
            key = (step_type, epoch, 0, call_order[0])
            call_order[0] += 1
            user = prompts.format_merge_user(skill_text, edits_list, meta_skill)
            output = client.complete(user, system=system, call_key=key)
            trace.record_model_call(step_type, system, user, output, call_key=key)
            return _parse_merge(output)
        # Multiple batches — merge each, then recurse
        merged_parts: list[dict] = []
        for batch in batches:
            if len(batch) == 1:
                merged_parts.append(batch[0])
                continue
            key = (step_type, epoch, 0, call_order[0])
            call_order[0] += 1
            user = prompts.format_merge_user(skill_text, batch, meta_skill)
            output = client.complete(user, system=system, call_key=key)
            trace.record_model_call(step_type, system, user, output, call_key=key)
            merged_parts.extend(_parse_merge(output))
        if len(merged_parts) > 1:
            return _merge_group(merged_parts, system, step_type)
        return merged_parts

    # Phase 1: failure edits
    merged_failures = _merge_group(
        failure_edits, prompts.MERGE_FAILURE_SYSTEM, "merge_failure",
    ) if failure_edits else []

    # Phase 2: success edits
    merged_successes = _merge_group(
        success_edits, prompts.MERGE_SUCCESS_SYSTEM, "merge_success",
    ) if success_edits else []

    # Single-group shortcut — aggregate.py:210-215
    if not merged_failures:
        return merged_successes
    if not merged_successes:
        return merged_failures

    # Phase 3: final merge — aggregate.py:217-275
    all_edits = merged_failures + merged_successes
    key = ("merge_final", epoch, 0, call_order[0])
    call_order[0] += 1
    user = prompts.format_merge_user(skill_text, all_edits, meta_skill)
    output = client.complete(
        user, system=prompts.MERGE_FINAL_SYSTEM, call_key=key,
    )
    trace.record_model_call(
        "merge_final", prompts.MERGE_FINAL_SYSTEM, user, output, call_key=key,
    )
    result = _parse_merge(output)
    if not result:
        # Concatenation fallback — aggregate.py:272-275
        result = merged_failures + merged_successes
    return result


# ── rank ──────────────────────────────────────────────────────────────────

def rank(
    client: Any,
    skill_text: str,
    merged_edits: list[dict[str, Any]],
    budget: int,
    meta_skill: str,
    epoch: int,
    trace: TraceRecorder,
) -> list[dict[str, Any]]:
    """Rank and select top edits up to the budget.

    Short-circuits if ``len(edits) <= budget`` (clip.py:54-55).
    Fallback: truncation ``edits[:budget]`` (clip.py:104-109).
    """
    if len(merged_edits) <= budget:
        return merged_edits

    call_key = ("rank", epoch, 0, 0)
    user = prompts.format_rank_user(skill_text, merged_edits, budget)
    output = client.complete(
        user, system=prompts.RANK_SYSTEM, call_key=call_key,
    )
    trace.record_model_call(
        "rank", prompts.RANK_SYSTEM, user, output, call_key=call_key,
    )

    selected = _parse_ranking(output, merged_edits, budget)
    trace.record_decision("rank", edit_budget=budget, selected_count=len(selected))
    return selected


# ── helpers ────────────────────────────────────────────────────────────────

def _chunk(items: list, size: int) -> list[list]:
    size = max(2, size)  # aggregate.py:96
    return [items[i:i + size] for i in range(0, len(items), size)]


def _parse_merge(output: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    edits = data.get("edits", [])
    return edits if isinstance(edits, list) else []


def _parse_ranking(
    output: str,
    edits: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """Extract selected edits by 0-based index — clip.py:80-100.

    Fallback: truncation — clip.py:104-109.
    """
    try:
        data = json.loads(output)
        indices = data.get("selected_indices", [])
    except (json.JSONDecodeError, TypeError):
        return edits[:budget]

    if not isinstance(indices, list):
        return edits[:budget]

    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for idx in indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(edits):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(edits[idx])
        if len(selected) >= budget:
            break

    if not selected:
        return edits[:budget]
    return selected
