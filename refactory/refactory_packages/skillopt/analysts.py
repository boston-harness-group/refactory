"""Failure / success analyst logic reproducing SkillOpt's reflect stage.

Split by ``passed`` (items.py:137 / items.py:127).
Shuffle failures with ``seed``, successes with ``seed + 1``
(reflect.py:562-563).
Chunk into groups of ``group_size`` (reflect.py:566-567).
"""

from __future__ import annotations

import json
import random
from typing import Any

from refactory_packages.skillopt import prompts
from refactory_packages.skillopt.trace import TraceRecorder


def reflect(
    client: Any,
    records: list[dict[str, Any]],
    skill_text: str,
    edit_budget: int,
    meta_skill: str,
    seed: int,
    group_size: int,
    epoch: int,
    trace: TraceRecorder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run failure and success analysts over train records.

    Returns ``(failure_edits, success_edits)`` — each a list of edit dicts.
    """
    failures = [r for r in records if not r.get("passed", False)]
    successes = [r for r in records if r.get("passed", False)]

    # Shuffle deterministically — reflect.py:562-563
    fail_rng = random.Random(seed)
    fail_rng.shuffle(failures)
    succ_rng = random.Random(seed + 1)
    succ_rng.shuffle(successes)

    failure_edits: list[dict[str, Any]] = []
    success_edits: list[dict[str, Any]] = []

    # Groups — reflect.py:566-567
    fail_groups = _chunk(failures, group_size)
    succ_groups = _chunk(successes, group_size)

    for gi, group in enumerate(fail_groups):
        call_key = ("analyst_failure", epoch, gi, 0)
        user = prompts.format_analyst_user(skill_text, group, edit_budget, meta_skill)
        output = client.complete(
            user, system=prompts.FAILURE_ANALYST_SYSTEM, call_key=call_key,
        )
        trace.record_model_call(
            "analyst_failure", prompts.FAILURE_ANALYST_SYSTEM, user, output,
            call_key=call_key,
        )
        trace.record_decision(
            "analyst_failure",
            group_index=gi,
            record_ids=[r.get("id", "") for r in group],
        )
        edits = _parse_edits(output, "failure")
        failure_edits.extend(edits)

    for gi, group in enumerate(succ_groups):
        call_key = ("analyst_success", epoch, gi, 0)
        user = prompts.format_analyst_user(skill_text, group, edit_budget, meta_skill)
        output = client.complete(
            user, system=prompts.SUCCESS_ANALYST_SYSTEM, call_key=call_key,
        )
        trace.record_model_call(
            "analyst_success", prompts.SUCCESS_ANALYST_SYSTEM, user, output,
            call_key=call_key,
        )
        trace.record_decision(
            "analyst_success",
            group_index=gi,
            record_ids=[r.get("id", "") for r in group],
        )
        edits = _parse_edits(output, "success")
        success_edits.extend(edits)

    return failure_edits, success_edits


def _chunk(items: list, size: int) -> list[list]:
    if not items:
        return []
    return [items[i:i + size] for i in range(0, len(items), size)]


def _parse_edits(output: str, source_type: str) -> list[dict[str, Any]]:
    """Extract edits from an analyst's JSON output."""
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    patch = data.get("patch", data)
    edits = patch.get("edits", [])
    if not isinstance(edits, list):
        return []
    for edit in edits:
        if isinstance(edit, dict):
            edit["source_type"] = source_type
    return edits
