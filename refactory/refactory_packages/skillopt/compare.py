"""Comparison report: A's recorded outputs vs B's SkillOpt run.

Loads A's ``out/`` directory (history.json, skills/, meta_skill/, slow_update/,
steps/) and B's trace, and produces a structured per-epoch comparison covering
all fields the brief requires.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ── loaders for A's outputs ───────────────────────────────────────────────


def load_a_history(a_out_dir: Path) -> list[dict[str, Any]]:
    """Read A's ``out/history.json``."""
    with open(a_out_dir / "history.json") as f:
        return json.load(f)


def load_a_skills(a_out_dir: Path) -> dict[int, str]:
    """Read A's ``out/skills/skill_v*.md`` → {version_number: text}."""
    skills: dict[int, str] = {}
    skills_dir = a_out_dir / "skills"
    if not skills_dir.exists():
        return skills
    for p in sorted(skills_dir.glob("skill_v*.md")):
        # e.g. skill_v0001.md → version 1
        stem = p.stem  # skill_v0001
        version_str = stem.split("_v")[-1]
        try:
            version = int(version_str)
        except ValueError:
            continue
        skills[version] = p.read_text()
    return skills


def load_a_meta_skill(a_out_dir: Path) -> dict[int, dict[str, Any]]:
    """Read A's ``out/meta_skill/epoch_*/meta_skill_result.json``."""
    result: dict[int, dict[str, Any]] = {}
    meta_dir = a_out_dir / "meta_skill"
    if not meta_dir.exists():
        return result
    for epoch_dir in sorted(meta_dir.glob("epoch_*")):
        epoch_str = epoch_dir.name.split("_")[-1]
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue
        result_file = epoch_dir / "meta_skill_result.json"
        if result_file.exists():
            with open(result_file) as f:
                result[epoch] = json.load(f)
    return result


def load_a_slow_update(a_out_dir: Path) -> dict[int, dict[str, Any]]:
    """Read A's ``out/slow_update/epoch_*/slow_result.json``."""
    result: dict[int, dict[str, Any]] = {}
    slow_dir = a_out_dir / "slow_update"
    if not slow_dir.exists():
        return result
    for epoch_dir in sorted(slow_dir.glob("epoch_*")):
        epoch_str = epoch_dir.name.split("_")[-1]
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue
        result_file = epoch_dir / "slow_result.json"
        if result_file.exists():
            with open(result_file) as f:
                result[epoch] = json.load(f)
    return result


def load_a_step_details(a_out_dir: Path, step: int) -> dict[str, Any]:
    """Read the step directory ``out/steps/step_NNNN/``."""
    step_dir = a_out_dir / "steps" / f"step_{step:04d}"
    details: dict[str, Any] = {}

    # rollout results
    rollout_file = step_dir / "rollout" / "results.jsonl"
    if rollout_file.exists():
        records = []
        with open(rollout_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        details["rollout_results"] = records

    # selection_eval results
    sel_file = step_dir / "selection_eval" / "results.jsonl"
    if sel_file.exists():
        records = []
        with open(sel_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        details["selection_eval_results"] = records

    # merged_patch
    merged_file = step_dir / "merged_patch.json"
    if merged_file.exists():
        with open(merged_file) as f:
            details["merged_patch"] = json.load(f)

    # ranked_edits
    ranked_file = step_dir / "ranked_edits.json"
    if ranked_file.exists():
        with open(ranked_file) as f:
            details["ranked_edits"] = json.load(f)

    # edit_apply_report
    apply_file = step_dir / "edit_apply_report.json"
    if apply_file.exists():
        with open(apply_file) as f:
            details["edit_apply_report"] = json.load(f)

    # candidate_skill
    skill_file = step_dir / "candidate_skill.md"
    if skill_file.exists():
        details["candidate_skill"] = skill_file.read_text()

    # patches directory
    patches_dir = step_dir / "patches"
    if patches_dir.exists():
        patch_files = sorted(patches_dir.glob("*.json"))
        details["patches"] = [p.name for p in patch_files]

    return details


# ── hashing ────────────────────────────────────────────────────────────────


def hash_skill(text: str) -> str:
    """16-char hex hash matching A's ``candidate_hash`` format in history.json."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── comparison ─────────────────────────────────────────────────────────────


def compare_runs(
    a_out_dir: Path,
    b_trace: list[dict[str, Any]],
    b_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare A's recorded run with B's.

    Returns ``{'per_epoch': [...], 'first_divergence': {...} | None,
    'summary': {...}}``.
    """
    a_history = load_a_history(a_out_dir)
    a_skills = load_a_skills(a_out_dir)
    a_meta = load_a_meta_skill(a_out_dir)
    a_slow = load_a_slow_update(a_out_dir)

    per_epoch: list[dict[str, Any]] = []
    first_div: dict[str, Any] | None = None

    for a_entry in a_history:
        epoch = a_entry["epoch"]
        step = a_entry["step"]

        # Find matching B entry
        b_entry = None
        for be in b_history:
            if be.get("epoch") == epoch and be.get("step") == step:
                b_entry = be
                break

        epoch_report: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "reflection_groups": {"a": None, "b": None, "match": None},
            "merged_edits": {"a": None, "b": None, "match": None},
            "ranked_edits": {"a": None, "b": None, "match": None},
            "edit_apply_outcomes": {"a": None, "b": None, "match": None},
            "candidate_skill_hash": {"a": None, "b": None, "match": None},
            "val_scores": {"a": None, "b": None, "match": None},
            "accept_reject": {"a": None, "b": None, "match": None},
            "slow_update": {"a": None, "b": None, "match": None},
            "meta_skill": {"a": None, "b": None, "match": None},
            "early_stop": {"a": None, "b": None, "match": None},
        }

        if b_entry is None:
            # B didn't reach this epoch/step at all.
            if first_div is None:
                first_div = {
                    "epoch": epoch,
                    "step": step,
                    "phase": "epoch_missing",
                    "field": "epoch_reached",
                    "a_value": True,
                    "b_value": False,
                    "call_key": None,
                }
            per_epoch.append(epoch_report)
            continue

        # Load A's step details
        a_step = load_a_step_details(a_out_dir, step)

        # -- reflection_groups
        a_patches_count = a_entry.get("n_patches")
        b_patches_count = b_entry.get("n_patches")
        a_fail = a_entry.get("n_failure_patches")
        a_succ = a_entry.get("n_success_patches")
        b_fail = b_entry.get("n_failure_patches")
        b_succ = b_entry.get("n_success_patches")
        a_refl = {"n_patches": a_patches_count, "n_failure": a_fail, "n_success": a_succ}
        b_refl = {"n_patches": b_patches_count, "n_failure": b_fail, "n_success": b_succ}
        refl_match = a_refl == b_refl
        epoch_report["reflection_groups"] = {"a": a_refl, "b": b_refl, "match": refl_match}
        if not refl_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "reflection",
                "field": "reflection_groups",
                "a_value": a_refl, "b_value": b_refl, "call_key": None,
            }

        # -- merged_edits
        a_merged = a_entry.get("n_edits_merged")
        b_merged = b_entry.get("n_edits_merged")
        merged_match = a_merged == b_merged
        epoch_report["merged_edits"] = {"a": a_merged, "b": b_merged, "match": merged_match}
        if not merged_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "merge",
                "field": "merged_edits",
                "a_value": a_merged, "b_value": b_merged, "call_key": None,
            }

        # -- ranked_edits
        a_ranked = a_entry.get("n_edits_ranked")
        b_ranked = b_entry.get("n_edits_ranked")
        ranked_match = a_ranked == b_ranked
        epoch_report["ranked_edits"] = {"a": a_ranked, "b": b_ranked, "match": ranked_match}
        if not ranked_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "rank",
                "field": "ranked_edits",
                "a_value": a_ranked, "b_value": b_ranked, "call_key": None,
            }

        # -- edit_apply_outcomes
        a_apply = a_entry.get("edit_apply_summary")
        b_apply = b_entry.get("edit_apply_summary")
        apply_match = a_apply == b_apply
        epoch_report["edit_apply_outcomes"] = {"a": a_apply, "b": b_apply, "match": apply_match}
        if not apply_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "edit_apply",
                "field": "edit_apply_outcomes",
                "a_value": a_apply, "b_value": b_apply, "call_key": None,
            }

        # -- candidate_skill_hash
        a_hash = a_entry.get("candidate_hash")
        b_hash_full = b_entry.get("candidate_hash", "")
        b_hash = b_hash_full[:16] if b_hash_full else b_hash_full
        hash_match = a_hash == b_hash
        epoch_report["candidate_skill_hash"] = {"a": a_hash, "b": b_hash, "match": hash_match}
        if not hash_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "skill_hash",
                "field": "candidate_skill_hash",
                "a_value": a_hash, "b_value": b_hash, "call_key": None,
            }

        # -- val_scores (selection_hard)
        a_val = a_entry.get("selection_hard")
        b_val = b_entry.get("selection_hard")
        val_match = a_val == b_val
        epoch_report["val_scores"] = {"a": a_val, "b": b_val, "match": val_match}
        if not val_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "val_evaluate",
                "field": "val_scores",
                "a_value": a_val, "b_value": b_val, "call_key": None,
            }

        # -- accept_reject
        a_action = a_entry.get("action")
        b_action = b_entry.get("action")
        action_match = a_action == b_action
        epoch_report["accept_reject"] = {"a": a_action, "b": b_action, "match": action_match}
        if not action_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "val_gate",
                "field": "accept_reject",
                "a_value": a_action, "b_value": b_action, "call_key": None,
            }

        # -- slow_update
        a_slow_ep = a_slow.get(epoch)
        b_slow_ep = b_entry.get("slow_update")
        slow_match = _comparable(a_slow_ep) == _comparable(b_slow_ep)
        epoch_report["slow_update"] = {"a": a_slow_ep is not None, "b": b_slow_ep is not None, "match": slow_match}
        if not slow_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "slow_update",
                "field": "slow_update",
                "a_value": a_slow_ep, "b_value": b_slow_ep, "call_key": None,
            }

        # -- meta_skill
        a_meta_ep = a_meta.get(epoch)
        b_meta_ep = b_entry.get("meta_skill")
        meta_match = _comparable(a_meta_ep) == _comparable(b_meta_ep)
        epoch_report["meta_skill"] = {"a": a_meta_ep is not None, "b": b_meta_ep is not None, "match": meta_match}
        if not meta_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "meta_skill",
                "field": "meta_skill",
                "a_value": a_meta_ep, "b_value": b_meta_ep, "call_key": None,
            }

        # -- early_stop
        a_early = False  # A completed all its epochs (or stopped later)
        b_early = b_entry.get("early_stop", False)
        early_match = a_early == b_early
        epoch_report["early_stop"] = {"a": a_early, "b": b_early, "match": early_match}
        if not early_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "early_stop",
                "field": "early_stop",
                "a_value": a_early, "b_value": b_early, "call_key": None,
            }

        per_epoch.append(epoch_report)

    # Summary
    total = len(per_epoch)
    matched = sum(
        1
        for ep in per_epoch
        if all(
            ep[field].get("match") is not False
            for field in (
                "reflection_groups", "merged_edits", "ranked_edits",
                "edit_apply_outcomes", "candidate_skill_hash", "val_scores",
                "accept_reject", "slow_update", "meta_skill", "early_stop",
            )
        )
    )

    return {
        "per_epoch": per_epoch,
        "first_divergence": first_div,
        "summary": {
            "total_epochs": total,
            "matched_epochs": matched,
            "diverged_epochs": total - matched,
        },
    }


def _comparable(value: Any) -> Any:
    """Reduce a value to a comparable form (handles None)."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def format_report(report: dict[str, Any]) -> str:
    """Render a human-readable text summary of the comparison report."""
    lines: list[str] = []
    lines.append("# SkillOpt Replay Comparison Report")
    lines.append("")

    summary = report.get("summary", {})
    lines.append(f"Total epochs compared: {summary.get('total_epochs', 0)}")
    lines.append(f"Matched epochs: {summary.get('matched_epochs', 0)}")
    lines.append(f"Diverged epochs: {summary.get('diverged_epochs', 0)}")
    lines.append("")

    first_div = report.get("first_divergence")
    if first_div:
        lines.append("## First Divergence")
        lines.append(f"  Epoch: {first_div.get('epoch')}")
        lines.append(f"  Step: {first_div.get('step')}")
        lines.append(f"  Phase: {first_div.get('phase')}")
        lines.append(f"  Field: {first_div.get('field')}")
        lines.append(f"  A value: {first_div.get('a_value')}")
        lines.append(f"  B value: {first_div.get('b_value')}")
        if first_div.get("call_key"):
            lines.append(f"  Call key: {first_div.get('call_key')}")
        lines.append("")
    else:
        lines.append("## No divergence found — A and B match completely.")
        lines.append("")

    lines.append("## Per-Epoch Detail")
    fields = [
        "reflection_groups", "merged_edits", "ranked_edits",
        "edit_apply_outcomes", "candidate_skill_hash", "val_scores",
        "accept_reject", "slow_update", "meta_skill", "early_stop",
    ]
    for ep in report.get("per_epoch", []):
        lines.append(f"### Epoch {ep['epoch']} (step {ep['step']})")
        for field in fields:
            info = ep.get(field, {})
            status = "✓" if info.get("match") else "✗" if info.get("match") is False else "?"
            lines.append(f"  {status} {field}: a={info.get('a')}, b={info.get('b')}")
        lines.append("")

    return "\n".join(lines)
