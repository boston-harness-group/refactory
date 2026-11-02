"""Failure / success analyst logic reproducing SkillOpt's reflect stage.

Split by ``passed`` (reflect.py:550-551).
Shuffle failures with the batch seed, successes with batch seed + 1
(reflect.py:553-554, ``_shuffle_for_minibatch`` at :471-482).
Chunk into groups of ``group_size`` (reflect.py:557-558).
Each group yields at most one patch (reflect.py:353-363, :450-457), and
patches without edits are dropped before the merge (trainer.py:157-186).
"""

from __future__ import annotations

import random
from typing import Any

from refactory_packages.skillopt import prompts
from refactory_packages.skillopt.env import SkillOptEnv
from refactory_packages.skillopt.json_utils import extract_json
from refactory_packages.skillopt.trace import TraceRecorder, ask


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
    *,
    env: SkillOptEnv | None = None,
    knobs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run failure and success analysts over train records.

    Returns ``(failure_patches, success_patches)``: one ``{"reasoning",
    "edits"}`` patch per analyst group that proposed edits, in group order.
    ``edit_budget`` is the analysts' constant L (envs/base.py:265 passes the
    adapter's fixed ``edit_budget``, not the scheduler's value).

    ``env`` is the mode's package as SkillOpt's environment: its
    ``analyst_error``/``analyst_success`` prompts (envs/base.py:304-330 load
    them with the env override) and its rendering of each record as a rollout
    result, given the ``knobs`` the records were rolled out with.
    """
    env = env if env is not None else SkillOptEnv()
    knobs = dict(knobs or {})
    failures = [r for r in records if not r.get("passed", False)]
    successes = [r for r in records if r.get("passed", False)]

    # Shuffle deterministically with the batch seed — reflect.py:553-554
    random.Random(seed).shuffle(failures)
    random.Random(seed + 1).shuffle(successes)

    failure_patches: list[dict[str, Any]] = []
    success_patches: list[dict[str, Any]] = []

    for kind, items, out in (
        ("failure", failures, failure_patches),
        ("success", successes, success_patches),
    ):
        step_type = f"analyst_{kind}"
        system = env.prompt("analyst_error" if kind == "failure" else "analyst_success")
        for gi, group in enumerate(_chunk(items, group_size)):
            call_key = (step_type, epoch, gi, 0)
            user = prompts.format_analyst_user(
                skill_text,
                [env.rollout_result(record, knobs) for record in group],
                edit_budget, meta_skill, failure=(kind == "failure"),
            )
            if not user:
                # No trajectory to show: SkillOpt makes no call (reflect.py:324-325).
                trace.record_decision(
                    step_type, group_index=gi,
                    record_ids=[r.get("id", "") for r in group], n_edits=0,
                )
                continue
            output = ask(client, trace, step_type, user, system=system, call_key=call_key)
            patch = _parse_patch(output, kind, edit_budget)
            parsed = extract_json(output) if isinstance(output, str) else None
            trace.record_decision(
                step_type,
                group_index=gi,
                record_ids=[r.get("id", "") for r in group],
                n_edits=len(patch["edits"]) if patch else 0,
                # Whether the reply carried a patch at all, edits or none: the
                # groups SkillOpt writes a minibatch_*.json for (reflect.py:646-650
                # on a reply with "patch", :366 / :460). Observability only.
                returned_patch=isinstance(parsed, dict) and bool(parsed) and "patch" in parsed,
            )
            if patch:
                out.append(patch)

    return failure_patches, success_patches


def _chunk(items: list, size: int) -> list[list]:
    """reflect.py:466-468 ``_split_minibatches``."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _parse_patch(output: Any, source_type: str, edit_budget: int) -> dict[str, Any] | None:
    """One analyst's patch, or ``None`` when it proposed no usable edits.

    reflect.py:353-363 / :450-457: the response must carry ``patch``; its
    edits are truncated to L (``truncate_payload``, update_modes.py:64-70).
    trainer.py:157-186 ``_normalise_patches``: a patch without edits is
    dropped; each edit gets ``source_type`` and ``support_count`` defaults,
    the latter ``max(batch_size, 1)``.
    """
    result = extract_json(output) if isinstance(output, str) else None
    if not isinstance(result, dict) or "patch" not in result:
        return None
    inner = result["patch"]
    if not isinstance(inner, dict):
        return None
    edits = inner.get("edits", [])
    if not isinstance(edits, list):
        return None
    if edit_budget >= 0 and len(edits) > edit_budget:
        inner["edits"] = edits = edits[:edit_budget]
    if not edits:
        return None
    try:
        support = max(int(result.get("batch_size", 0) or 0), 1)
    except (TypeError, ValueError):
        support = 1
    for edit in edits:
        if isinstance(edit, dict):
            edit.setdefault("source_type", source_type)
            edit.setdefault("support_count", support)
    return inner
