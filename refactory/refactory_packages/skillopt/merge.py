"""Failure-first hierarchical merge and LLM rank.

Reproduces ``aggregate.py:28-275`` (hierarchical merge of analyst patches)
and ``clip.py:25-109`` (rank and select), patch update mode.
"""

from __future__ import annotations

from typing import Any

from refactory_packages.skillopt import prompts
from refactory_packages.skillopt.json_utils import extract_json
from refactory_packages.skillopt.trace import TraceRecorder, ask

# ── hierarchical merge ────────────────────────────────────────────────────

def hierarchical_merge(
    client: Any,
    skill_text: str,
    failure_patches: list[dict[str, Any]],
    success_patches: list[dict[str, Any]],
    meta_skill: str,
    epoch: int,
    trace: TraceRecorder,
    merge_batch_size: int = 8,
) -> dict[str, Any]:
    """Failure-first hierarchical merge of analyst patches — aggregate.py:156-275.

    Phase 1: merge the failure patches hierarchically (aggregate.py:195-199).
    Phase 2: merge the success patches hierarchically (aggregate.py:201-205).
    Phase 3: final merge with failure priority (aggregate.py:217-270), falling
    back to failure edits then success edits (aggregate.py:272-275).

    Returns the merged patch dict (``reasoning`` + ``edits``).
    """
    failure_merged = _merge_group(
        client, skill_text, failure_patches, prompts.MERGE_FAILURE_SYSTEM,
        "merge_failure", meta_skill, epoch, trace, merge_batch_size,
    )
    success_merged = _merge_group(
        client, skill_text, success_patches, prompts.MERGE_SUCCESS_SYSTEM,
        "merge_success", meta_skill, epoch, trace, merge_batch_size,
    )

    f_edits = patch_edits(failure_merged)
    s_edits = patch_edits(success_merged)

    # Single-group shortcuts — aggregate.py:210-215
    if not f_edits and not s_edits:
        return {"reasoning": "no updates from either group", "edits": []}
    if not s_edits:
        return failure_merged
    if not f_edits:
        return success_merged

    key = ("merge_final", epoch, 0, 0)
    user = prompts.format_merge_final_user(
        skill_text, failure_merged, success_merged, len(f_edits), len(s_edits), meta_skill,
    )
    output = ask(client, trace, "merge_final", user, system=prompts.MERGE_FINAL_SYSTEM, call_key=key)
    final = _parse_merge(output)
    if final is not None:
        trace.record_decision("merge_final", fallback=False, n_edits=len(final["edits"]))
        return final
    # Fallback — aggregate.py:272-275
    trace.record_decision("merge_final", fallback=True, n_edits=len(f_edits) + len(s_edits))
    return {"reasoning": "fallback: failure first, then success", "edits": f_edits + s_edits}


def _merge_group(
    client: Any,
    skill_text: str,
    patches: list[dict[str, Any]],
    system: str,
    step_type: str,
    meta_skill: str,
    epoch: int,
    trace: TraceRecorder,
    batch_size: int,
) -> dict[str, Any]:
    """Merge N patches level by level until one remains — aggregate.py:79-151."""
    if not patches:
        return {"reasoning": "no patches", "edits": []}  # aggregate.py:97-98
    if len(patches) == 1:
        return patches[0]  # aggregate.py:99-100 — no model call

    batch_size = max(2, batch_size)  # aggregate.py:102
    current = list(patches)
    level = 0
    while len(current) > 1:
        level += 1
        next_level: list[dict[str, Any]] = []
        for bi, start in enumerate(range(0, len(current), batch_size)):
            batch = current[start:start + batch_size]
            if len(batch) == 1:
                next_level.append(batch[0])  # aggregate.py:123-124
                continue
            next_level.append(_merge_batch(
                client, skill_text, batch, system, step_type, meta_skill,
                epoch, trace, level, bi,
            ))
        current = next_level
    return current[0]


def _merge_batch(
    client: Any,
    skill_text: str,
    patches: list[dict[str, Any]],
    system: str,
    step_type: str,
    meta_skill: str,
    epoch: int,
    trace: TraceRecorder,
    level: int,
    batch_index: int,
) -> dict[str, Any]:
    """Merge one batch of patches into one — aggregate.py:28-76."""
    key = (step_type, epoch, batch_index, level)
    user = prompts.format_merge_user(skill_text, patches, meta_skill, level)
    output = ask(client, trace, step_type, user, system=system, call_key=key)
    merged = _parse_merge(output)
    if merged is not None:
        for edit in merged["edits"]:
            edit["merge_level"] = level  # aggregate.py:57-58
        trace.record_decision(
            step_type, level=level, batch_index=batch_index,
            n_patches=len(patches), fallback=False, n_edits=len(merged["edits"]),
        )
        return merged
    # Fallback: concatenate every edit, never drop one — aggregate.py:70-76
    all_edits: list[dict[str, Any]] = []
    for patch in patches:
        for edit in patch_edits(patch):
            edit.setdefault("merge_level", level)
            all_edits.append(edit)
    trace.record_decision(
        step_type, level=level, batch_index=batch_index,
        n_patches=len(patches), fallback=True, n_edits=len(all_edits),
    )
    return {"reasoning": "fallback concatenation", "edits": all_edits}


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
    """Rank and select top edits up to the budget — clip.py:25-109.

    Short-circuits if ``len(edits) <= budget`` (clip.py:54-55).
    Fallback: truncation ``edits[:budget]`` (clip.py:104-109).
    """
    if len(merged_edits) <= budget:
        return merged_edits

    call_key = ("rank", epoch, 0, 0)
    user = prompts.format_rank_user(skill_text, merged_edits, budget, meta_skill)
    output = ask(client, trace, "rank", user, system=prompts.RANK_SYSTEM, call_key=call_key)

    selected = _parse_ranking(output, merged_edits, budget)
    trace.record_decision("rank", edit_budget=budget, selected_count=len(selected))
    return selected


# ── helpers ────────────────────────────────────────────────────────────────

def patch_edits(patch: dict[str, Any] | None) -> list[dict[str, Any]]:
    """update_modes.py:52-56 ``get_payload_items`` (patch mode)."""
    if not isinstance(patch, dict):
        return []
    edits = patch.get("edits", [])
    return edits if isinstance(edits, list) else []


def _parse_merge(output: Any) -> dict[str, Any] | None:
    """A usable merge response or ``None`` — aggregate.py:53-59 and :251-260.

    Usable means ``edits`` is a list of dicts (an empty list is usable).
    """
    merged = extract_json(output) if isinstance(output, str) else None
    items = merged.get("edits") if isinstance(merged, dict) else None
    if isinstance(items, list) and all(isinstance(e, dict) for e in items):
        return merged
    return None


def _parse_ranking(
    output: Any,
    edits: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """Extract selected edits by 0-based index — clip.py:79-100.

    Fallback: truncation — clip.py:104-109.
    """
    result = extract_json(output) if isinstance(output, str) else None
    if not result or not isinstance(result, dict) or "selected_indices" not in result:
        return edits[:budget]
    indices = result["selected_indices"]
    if not isinstance(indices, list):
        return edits[:budget]

    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(edits) and idx not in seen:
            selected.append(edits[idx])
            seen.add(idx)
        if len(selected) >= budget:
            break

    if not selected:
        return edits[:budget]
    return selected
