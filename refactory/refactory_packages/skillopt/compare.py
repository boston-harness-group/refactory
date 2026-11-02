"""Comparison report: A's recorded outputs vs B's SkillOpt run.

Loads A's ``out/`` directory (history.json, skills/, meta_skill/, slow_update/,
steps/) and B's history, and produces a structured per-epoch comparison covering
all fields the brief requires: reflection groups (ids and order), merged edits,
ranked edits, per-edit apply outcomes, candidate skill hash, val scores,
accept/reject, slow update, meta skill, early stop.
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
        details["rollout_ids"] = [r.get("id") for r in records]

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


def _comparable(value: Any) -> Any:
    """Reduce a value to a comparable form (handles None)."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def compare_runs(
    a_out_dir: Path,
    b_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare A's recorded run with B's replay output.

    Returns ``{'per_epoch': [...], 'first_divergence': {...} | None,
    'summary': {...}}``.
    """
    a_history = load_a_history(a_out_dir)
    a_skills = load_a_skills(a_out_dir)
    a_meta = load_a_meta_skill(a_out_dir)
    a_slow = load_a_slow_update(a_out_dir)

    # A's config tells us num_epochs
    a_config_path = a_out_dir / "config.json"
    a_num_epochs = None
    if a_config_path.exists():
        with open(a_config_path) as f:
            a_cfg = json.load(f)
        a_num_epochs = a_cfg.get("num_epochs")

    # A early-stopped if it has fewer entries than num_epochs
    a_actually_early_stopped = (
        a_num_epochs is not None and len(a_history) < a_num_epochs
    )

    per_epoch: list[dict[str, Any]] = []
    first_div: dict[str, Any] | None = None

    # Index B history by (epoch, step)
    b_by_epoch_step: dict[tuple[int, int], dict[str, Any]] = {}
    for be in b_history:
        key = (be.get("epoch", -1), be.get("step", -1))
        b_by_epoch_step[key] = be

    # Walk A's history
    for a_entry in a_history:
        epoch = a_entry["epoch"]
        step = a_entry["step"]

        b_entry = b_by_epoch_step.get((epoch, step))

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

        # Load A's step details for structural comparison
        a_step = load_a_step_details(a_out_dir, step)

        # ── reflection_groups: compare IDs and order ──
        a_refl = {
            "n_patches": a_entry.get("n_patches"),
            "n_failure": a_entry.get("n_failure_patches"),
            "n_success": a_entry.get("n_success_patches"),
            "rollout_ids": a_step.get("rollout_ids"),
            "patch_files": a_step.get("patches"),
        }
        b_refl = {
            "n_patches": b_entry.get("n_patches"),
            "n_failure": b_entry.get("n_failure_patches"),
            "n_success": b_entry.get("n_success_patches"),
            "rollout_ids": b_entry.get("rollout_ids"),
        }
        # Compare on available data: IDs and order if present, else counts
        if a_refl.get("rollout_ids") and b_refl.get("rollout_ids"):
            refl_match = a_refl["rollout_ids"] == b_refl["rollout_ids"]
        elif (a_refl["n_patches"] is not None
              and b_refl["n_patches"] is not None):
            refl_match = (
                a_refl["n_patches"] == b_refl["n_patches"]
                and a_refl["n_failure"] == b_refl["n_failure"]
                and a_refl["n_success"] == b_refl["n_success"]
            )
        else:
            refl_match = None  # insufficient data
        epoch_report["reflection_groups"] = {
            "a": a_refl, "b": b_refl, "match": refl_match,
        }
        if refl_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "reflection",
                "field": "reflection_groups",
                "a_value": a_refl, "b_value": b_refl, "call_key": None,
            }

        # ── merged_edits ──
        a_merged = a_step.get("merged_patch")
        a_merged_count = a_entry.get("n_edits_merged")
        b_merged = b_entry.get("merged_patch")
        b_merged_count = b_entry.get("n_edits_merged")
        if a_merged is not None and b_merged is not None:
            merged_match = _comparable(a_merged) == _comparable(b_merged)
        elif a_merged_count is not None and b_merged_count is not None:
            merged_match = a_merged_count == b_merged_count
        else:
            merged_match = None
        epoch_report["merged_edits"] = {
            "a": {"count": a_merged_count, "detail": a_merged is not None},
            "b": {"count": b_merged_count, "detail": b_merged is not None},
            "match": merged_match,
        }
        if merged_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "merge",
                "field": "merged_edits",
                "a_value": a_merged_count, "b_value": b_merged_count,
                "call_key": None,
            }

        # ── ranked_edits ──
        a_ranked = a_step.get("ranked_edits")
        a_ranked_count = a_entry.get("n_edits_ranked")
        b_ranked = b_entry.get("ranked_edits")
        b_ranked_count = b_entry.get("n_edits_ranked")
        if a_ranked is not None and b_ranked is not None:
            ranked_match = _comparable(a_ranked) == _comparable(b_ranked)
        elif a_ranked_count is not None and b_ranked_count is not None:
            ranked_match = a_ranked_count == b_ranked_count
        else:
            ranked_match = None
        epoch_report["ranked_edits"] = {
            "a": {"count": a_ranked_count, "detail": a_ranked is not None},
            "b": {"count": b_ranked_count, "detail": b_ranked is not None},
            "match": ranked_match,
        }
        if ranked_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "rank",
                "field": "ranked_edits",
                "a_value": a_ranked_count, "b_value": b_ranked_count,
                "call_key": None,
            }

        # ── edit_apply_outcomes ──
        a_apply = a_step.get("edit_apply_report")
        a_apply_summary = a_entry.get("edit_apply_summary")
        b_apply = b_entry.get("edit_apply_report")
        b_apply_summary = b_entry.get("edit_apply_summary")
        if a_apply and b_apply:
            a_statuses = [e.get("status") for e in a_apply]
            b_statuses = [e.get("status") for e in b_apply]
            apply_match = a_statuses == b_statuses
            epoch_report["edit_apply_outcomes"] = {
                "a": {"summary": a_apply_summary, "per_edit": a_statuses},
                "b": {"summary": b_apply_summary, "per_edit": b_statuses},
                "match": apply_match,
            }
        elif a_apply_summary is not None and b_apply_summary is not None:
            apply_match = a_apply_summary == b_apply_summary
            epoch_report["edit_apply_outcomes"] = {
                "a": a_apply_summary, "b": b_apply_summary, "match": apply_match,
            }
        else:
            apply_match = None
            epoch_report["edit_apply_outcomes"] = {
                "a": a_apply_summary, "b": b_apply_summary, "match": apply_match,
            }
        if apply_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "edit_apply",
                "field": "edit_apply_outcomes",
                "a_value": a_apply_summary, "b_value": b_apply_summary,
                "call_key": None,
            }

        # ── candidate_skill_hash ──
        a_hash = a_entry.get("candidate_hash")
        b_hash_full = b_entry.get("candidate_hash", "")
        b_hash = b_hash_full[:16] if b_hash_full else b_hash_full

        a_skill_text = a_skills.get(step) or (a_step.get("candidate_skill") or "")
        if a_skill_text:
            a_hash_computed = hash_skill(a_skill_text)
            epoch_report["candidate_skill_hash"] = {
                "a": a_hash,
                "a_computed_from_file": a_hash_computed,
                "b": b_hash,
                "match": a_hash == b_hash,
                "a_consistent": a_hash == a_hash_computed,
            }
        else:
            epoch_report["candidate_skill_hash"] = {
                "a": a_hash, "b": b_hash, "match": a_hash == b_hash if a_hash and b_hash else None,
            }
        hash_match = a_hash == b_hash if a_hash and b_hash else None
        if hash_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "skill_hash",
                "field": "candidate_skill_hash",
                "a_value": a_hash, "b_value": b_hash, "call_key": None,
            }

        # ── val_scores (selection_hard) ──
        a_val = a_entry.get("selection_hard")
        b_val = b_entry.get("selection_hard")
        if a_val is not None and b_val is not None:
            val_match = a_val == b_val
        else:
            val_match = None
        epoch_report["val_scores"] = {"a": a_val, "b": b_val, "match": val_match}
        if val_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "val_evaluate",
                "field": "val_scores",
                "a_value": a_val, "b_value": b_val, "call_key": None,
            }

        # ── accept_reject ──
        a_action = a_entry.get("action")
        b_action = b_entry.get("action")
        if a_action is not None and b_action is not None:
            action_match = a_action == b_action
        else:
            action_match = None
        epoch_report["accept_reject"] = {
            "a": a_action, "b": b_action, "match": action_match,
        }
        if action_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "val_gate",
                "field": "accept_reject",
                "a_value": a_action, "b_value": b_action, "call_key": None,
            }

        # ── slow_update ──
        a_slow_ep = a_slow.get(epoch)
        b_slow_ep = b_entry.get("slow_update")
        if a_slow_ep is not None and b_slow_ep is not None:
            slow_match = _comparable(a_slow_ep) == _comparable(b_slow_ep)
        else:
            slow_match = None
        epoch_report["slow_update"] = {
            "a": a_slow_ep is not None, "b": b_slow_ep is not None, "match": slow_match,
        }
        if slow_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "slow_update",
                "field": "slow_update",
                "a_value": a_slow_ep, "b_value": b_slow_ep, "call_key": None,
            }

        # ── meta_skill ──
        a_meta_ep = a_meta.get(epoch)
        b_meta_ep = b_entry.get("meta_skill")
        if a_meta_ep is not None and b_meta_ep is not None:
            meta_match = _comparable(a_meta_ep) == _comparable(b_meta_ep)
        else:
            meta_match = None
        epoch_report["meta_skill"] = {
            "a": a_meta_ep is not None, "b": b_meta_ep is not None, "match": meta_match,
        }
        if meta_match is False and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "meta_skill",
                "field": "meta_skill",
                "a_value": a_meta_ep, "b_value": b_meta_ep, "call_key": None,
            }

        # ── early_stop ──
        is_last_a_entry = (a_entry is a_history[-1])
        a_early = a_actually_early_stopped if is_last_a_entry else False
        b_early = b_entry.get("early_stop", False)
        early_match = a_early == b_early
        epoch_report["early_stop"] = {
            "a": a_early, "b": b_early, "match": early_match,
        }
        if not early_match and first_div is None:
            first_div = {
                "epoch": epoch, "step": step, "phase": "early_stop",
                "field": "early_stop",
                "a_value": a_early, "b_value": b_early, "call_key": None,
            }

        per_epoch.append(epoch_report)

    # Check for extra B epochs not in A's history
    a_epoch_steps = {(e["epoch"], e["step"]) for e in a_history}
    for be in b_history:
        key = (be.get("epoch", -1), be.get("step", -1))
        if key not in a_epoch_steps:
            extra_report: dict[str, Any] = {
                "epoch": be.get("epoch"),
                "step": be.get("step"),
                "extra_b_epoch": True,
                "reflection_groups": {"a": None, "b": None, "match": False},
                "merged_edits": {"a": None, "b": None, "match": False},
                "ranked_edits": {"a": None, "b": None, "match": False},
                "edit_apply_outcomes": {"a": None, "b": None, "match": False},
                "candidate_skill_hash": {"a": None, "b": None, "match": False},
                "val_scores": {"a": None, "b": None, "match": False},
                "accept_reject": {"a": None, "b": None, "match": False},
                "slow_update": {"a": None, "b": None, "match": False},
                "meta_skill": {"a": None, "b": None, "match": False},
                "early_stop": {"a": None, "b": None, "match": False},
            }
            per_epoch.append(extra_report)
            if first_div is None:
                first_div = {
                    "epoch": be.get("epoch"),
                    "step": be.get("step"),
                    "phase": "extra_b_epoch",
                    "field": "epoch_count",
                    "a_value": False,
                    "b_value": True,
                    "call_key": None,
                }

    # Summary
    total = len(per_epoch)
    fields_to_check = (
        "reflection_groups", "merged_edits", "ranked_edits",
        "edit_apply_outcomes", "candidate_skill_hash", "val_scores",
        "accept_reject", "slow_update", "meta_skill", "early_stop",
    )
    matched = sum(
        1
        for ep in per_epoch
        if all(
            ep[field].get("match") is not False
            for field in fields_to_check
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
        extra = " (extra B epoch)" if ep.get("extra_b_epoch") else ""
        lines.append(f"### Epoch {ep['epoch']} (step {ep['step']}){extra}")
        for field in fields:
            info = ep.get(field, {})
            match_val = info.get("match")
            if match_val is True:
                status = "✓"
            elif match_val is False:
                status = "✗"
            else:
                status = "?"
            lines.append(f"  {status} {field}: a={info.get('a')}, b={info.get('b')}")
        lines.append("")

    return "\n".join(lines)
