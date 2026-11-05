"""CLI entry point for offline SkillOpt replay.

Drives **arm B** — the ``SkillOptLoop`` strategy ``strategies.resolve("skillopt")``
finds by package discovery — on the LiveMath solver graph with A's seed, A's
config and A's recorded model outputs in place of model calls.  Produces a
comparison report naming the first point where B diverges from A.

Every code path under test is arm B's own:
  - ``strategies.create("skillopt", "livemath/solver", …)`` builds the Loop and
    attaches ``SkillOptLoop``, exactly as ``--strategy skillopt`` does.
  - ``livemath/mode.py`` + ``livemath/ops.py`` build the solver prompts.
  - ``skillopt/loop.py``, ``analysts.py``, ``merge.py`` run the search, so a bug
    in any of them shows up here.
  - ``skillopt/replay.py`` feeds A's recorded outputs to the model seam.

Usage::

    python -m refactory_packages.skillopt.replay_runner \\
        --recording-dir ~/skillopt/runs/skillopt-train-20261023-e10-s42 \\
        --split-dir ~/skillopt/data/livemathematicianbench_split \\
        --data-path ~/skillopt/data/livemath_raw \\
        --output-dir /tmp/replay_s42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from refactory_dsh import strategies
from refactory_dsh.loop import LoopConfig

from refactory_packages.livemath.manifest import build_parts_manifest
from refactory_packages.skillopt import prompts
from refactory_packages.skillopt.compare import compare_runs, format_report
from refactory_packages.skillopt.replay import (
    CallMatcher,
    RecordedPatchOrder,
    RecordedRolloutOrder,
    late_answer_lines,
    ReplayClient,
    ReplayExhausted,
    ReplayMismatch,
)

#: The mode the replay drives: the LiveMathematicianBench one-node solver.
MODE_ID = "livemath/solver"

#: The strategy name package discovery resolves to ``SkillOptLoop``.
STRATEGY = "skillopt"

#: Env var ``LiveMathTask`` reads its item corpus from (tasks.py:171).
DATA_PATH_ENV = "LIVEMATH_DATA_PATH"

#: The ops arm B's SkillOpt search needs (``SkillOptLoop.required_ops``).
ALLOWED_OPS = ("set_knob",)
#: The LiveMath solver knob the skill text lives in (livemath/mode.py `knobs`).
SKILL_KNOB = "skill"


# ── config mapping ─────────────────────────────────────────────────────────

#: A's ``config.json`` key → the ``SkillOptLoop.configure`` option it drives
#: (``skillopt/loop.py:73``).  Everything the replay configures comes from this
#: map applied to A's own file, never from a literal in this module.
A_CONFIG_TO_STRATEGY_OPTION = {
    "num_epochs": "max_epochs",
    "early_stop_patience": "early_stop_patience",
    "early_stop_at_full_val": "early_stop_at_full_val",
    "edit_budget": "edit_budget_max",
    "min_edit_budget": "edit_budget_min",
    "seed": "seed",
    "slow_update_samples": "slow_update_samples",
    "minibatch_size": "group_size",
    "merge_batch_size": "merge_batch_size",
}

#: A's config keys mapped only when the recording has them: ``eval_test``
#: (trainer.py:2241), the post-training evaluation -- the final skill's val check,
#: then the initial, best and final skills on test. Unset, B runs it when the parts
#: manifest has a test part, as A's default ``eval_test: true`` does.
A_CONFIG_OPTIONAL = {"eval_test": "eval_test"}

#: A's settings arm B has no option for, with the value arm B always behaves as.
#: A recording that differs on one of these cannot be replayed faithfully, so the
#: runner reports it rather than quietly comparing two different algorithms.
A_CONFIG_REQUIRED = {
    "use_slow_update": True,
    "use_meta_skill": True,
    "skill_update_mode": "patch",
    "failure_only": False,
    "use_gate": True,
    "steps_per_epoch": 1,
    "batches_per_epoch": 1,
    "accumulation": 1,
    # B evaluates the whole val and test parts; 0 is A's "every item".
    "sel_env_num": 0,
    "test_env_num": 0,
}

#: A's summary.json post-training figures (trainer.py:2480-2515) that B's strategy
#: report carries under the same names (``SkillOptLoop.report``).
A_TEST_EVAL_FIELDS = ("final_selection_hard", "baseline_test_hard", "test_hard", "final_test_hard")


def load_a_config(a_out_dir: Path) -> dict[str, Any]:
    """Map A's ``config.json`` to ``SkillOptLoop`` strategy options.

    A's key → B's option, from :data:`A_CONFIG_TO_STRATEGY_OPTION`, plus
    ``analyst_edit_budget``: the analysts' L is the adapter's constant
    ``edit_budget`` rather than the schedule's value (envs/base.py:265).
    """
    with open(a_out_dir / "config.json") as f:
        raw = json.load(f)

    options: dict[str, Any] = {
        option: raw[key] for key, option in A_CONFIG_TO_STRATEGY_OPTION.items()
    }
    options["analyst_edit_budget"] = raw["edit_budget"]
    options.update({option: raw[key] for key, option in A_CONFIG_OPTIONAL.items() if key in raw})
    return options


def unsupported_a_settings(a_out_dir: Path) -> list[str]:
    """A's settings arm B cannot be configured to match, as readable lines.

    A's ``exec_timeout`` is checked against the LiveMath solver's own
    ``solve_timeout_s`` rather than set from it: B's default is meant to be
    A's (livemath/mode.py ``SOLVE_TIMEOUT_S``), and a recording run with
    another value times items out at a different point.
    """
    from refactory_packages.livemath.mode import Solver

    with open(a_out_dir / "config.json") as f:
        raw = json.load(f)
    problems: list[str] = []
    if "exec_timeout" in raw and int(raw["exec_timeout"]) != int(Solver.solve_timeout_s):
        problems.append(
            f"A ran with exec_timeout={raw['exec_timeout']!r}; arm B's LiveMath solver "
            f"times out at {Solver.solve_timeout_s}s, so the two runs are not comparable"
        )
    for key, required in A_CONFIG_REQUIRED.items():
        if key in raw and raw[key] != required:
            problems.append(
                f"A ran with {key}={raw[key]!r}; arm B always behaves as "
                f"{key}={required!r}, so the two runs are not comparable"
            )
    return problems


# ── data loading ───────────────────────────────────────────────────────────


def _skill_init_of(recording_dir: Path) -> str:
    """A's configured ``skill_init`` path, for the refusal message."""
    config_path = recording_dir / "out" / "config.json"
    if not config_path.exists():
        return ""
    with open(config_path) as handle:
        return str(json.load(handle).get("skill_init", ""))


def _load_initial_skill(recording_dir: Path) -> str:
    """Load the initial skill A started from.

    Preferred source is the recording's own ``out/skills/skill_v0000.md``: SkillOpt
    writes the seed there (trainer.py saves each version), so it is the authoritative
    record of what this run actually started with, and it survives the env file moving
    or changing after the run. ``config.json``'s ``skill_init`` path is the fallback.
    """
    config_path = recording_dir / "out" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        skill_init = cfg.get("skill_init", "")
        if skill_init:
            # Try absolute path first
            resolved = Path(skill_init)
            if resolved.is_absolute() and resolved.exists():
                return resolved.read_text()
            # SkillOpt resolves relative to its own project root ~/skillopt
            for base in [Path("/home/lab/skillopt"), Path.home() / "skillopt"]:
                candidate = base / skill_init
                if candidate.exists():
                    return candidate.read_text()
    return ""


def _recorded_seed_skill(recording_dir: Path) -> str:
    """The seed skill as the recording itself stored it (``out/skills/skill_v0000.md``)."""
    recorded = recording_dir / "out" / "skills" / "skill_v0000.md"
    return recorded.read_text() if recorded.exists() else ""


def _load_train_items(data_dir: Path) -> list[dict[str, Any]]:
    """Load training items from the split directory."""
    items_file = data_dir / "train" / "items.json"
    if items_file.exists():
        with open(items_file) as f:
            return json.load(f)
    raise FileNotFoundError(f"No training items found at {items_file}")


def _load_val_items(data_dir: Path) -> list[dict[str, Any]]:
    """Load validation items from the split directory."""
    items_file = data_dir / "val" / "items.json"
    if items_file.exists():
        with open(items_file) as f:
            return json.load(f)
    return []


def _resolve_split_dir(a_out_dir: Path) -> Path | None:
    """A's ``split_dir``, resolved against SkillOpt's own project root."""
    config_path = a_out_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    split_dir = cfg.get("split_dir", "")
    if not split_dir:
        return None
    direct = Path(split_dir)
    if direct.is_absolute() and direct.exists():
        return direct
    for base in [Path("/home/lab/skillopt"), Path.home() / "skillopt"]:
        candidate = base / split_dir
        if candidate.exists():
            return candidate
    return None


# ── B's history, from B's own trace ────────────────────────────────────────

#: A's ``history.json`` ``action`` for the epochs that never reach the val gate
#: (trainer.py writes ``skip_no_patches`` when the analysts proposed nothing).
_SKIP_ACTION = "skip_no_patches"


def read_trace(runs_root: Path) -> list[dict[str, Any]]:
    """Read ``<runs_root>/skillopt_trace.jsonl``, the file ``SkillOptLoop`` writes.

    The file's existence is itself the evidence that arm B's strategy ran: no
    other code in the package writes it.
    """
    path = Path(runs_root) / "skillopt_trace.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _first(decisions: list[dict[str, Any]], step_type: str) -> dict[str, Any] | None:
    for decision in decisions:
        if decision.get("step_type") == step_type:
            return decision
    return None


def _all(decisions: list[dict[str, Any]], step_type: str) -> list[dict[str, Any]]:
    return [d for d in decisions if d.get("step_type") == step_type]


def _apply_summary(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    """A's ``edit_apply_summary`` (trainer.py:1520-1530) over B's outcomes."""
    return {
        "total": len(outcomes),
        "applied": sum(
            1 for row in outcomes if str(row.get("status", "")).startswith("applied")
        ),
        "skipped": sum(
            1 for row in outcomes if str(row.get("status", "")).startswith("skipped")
        ),
        "errors": sum(1 for row in outcomes if row.get("status") == "error"),
    }


def _patch_files(
    failure: list[dict[str, Any]], success: list[dict[str, Any]],
) -> list[str]:
    """The names A gives the groups that produced a patch (reflect.py:583, :592).

    A writes one file per analyst group whose reply carried a patch, named by
    the group's index, and ``load_a_step_details`` lists them sorted.  B's analyst
    decisions carry the same index, so the two listings compare directly.
    """
    names = [f"minibatch_fail_{int(d['group_index']):03d}.json" for d in failure]
    names += [f"minibatch_succ_{int(d['group_index']):03d}.json" for d in success]
    return sorted(names)


#: The slow-update and meta-skill fields both runs record
#: (A: slow_update/epoch_NN/slow_result.json, meta_skill/epoch_NN/meta_skill_result.json;
#: B: its ``slow_update`` / ``meta_skill`` decisions). Compared on the keys both have.
_SLOW_UPDATE_FIELDS = (
    "action", "reasoning", "slow_update_content", "prev_hard", "curr_hard", "candidate_hash",
)
_META_SKILL_FIELDS = ("action", "reasoning", "meta_skill_content")


def b_history_from_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the ``compare_runs`` B-side history from arm B's own trace.

    One entry per epoch arm B ran, keyed ``(epoch, step)``.  SkillOpt's LiveMath
    config has ``steps_per_epoch: 1`` and ``step_in_epoch: 0`` in every history
    entry, so A's step number equals its epoch number and ``step = epoch`` here;
    :data:`A_CONFIG_REQUIRED` refuses a recording where that does not hold.

    Every value comes from a decision arm B recorded.  A field arm B never
    recorded is left ``None``: ``compare_runs`` treats "A has data, B has None"
    as a non-match, which is the honest answer — nothing here is inferred to
    make a field agree.
    """
    decisions = [entry for entry in trace if entry.get("type") == "decision"]
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for decision in decisions:
        by_epoch.setdefault(int(decision.get("epoch", 0) or 0), []).append(decision)

    # The baseline val score of the seed skill, so the first accept can be told
    # apart from an accept that is not a new best (A's two ``action`` values).
    best: float | None = None
    initial = _first(by_epoch.get(0, []), "initial_val")
    if initial is not None and initial.get("score") is not None:
        best = float(initial["score"])

    history: list[dict[str, Any]] = []
    for epoch in sorted(e for e in by_epoch if e >= 1):
        epoch_decisions = by_epoch[epoch]
        entry: dict[str, Any] = {"epoch": epoch, "step": epoch}

        # ── rollout: the Run's per-item records, reduced in the trace ──
        rollout = _first(epoch_decisions, "train_rollout")
        entry["rollout_hard"] = rollout.get("rollout_hard") if rollout else None
        entry["rollout_ids"] = rollout.get("rollout_ids") if rollout else None
        entry["rollout_n"] = rollout.get("n_records") if rollout else None

        # ── analyst groups ──
        failure = _all(epoch_decisions, "analyst_failure")
        success = _all(epoch_decisions, "analyst_success")
        if failure or success:
            with_edits_f = [d for d in failure if int(d.get("n_edits") or 0) > 0]
            with_edits_s = [d for d in success if int(d.get("n_edits") or 0) > 0]
            entry["n_failure_patches"] = len(with_edits_f)
            entry["n_success_patches"] = len(with_edits_s)
            entry["n_patches"] = len(with_edits_f) + len(with_edits_s)
            # A writes a file for every group whose reply carried a patch, with
            # edits or without; traces older than ``returned_patch`` fall back
            # to the groups with edits.
            entry["patch_files"] = _patch_files(
                [d for d in failure if d.get("returned_patch", int(d.get("n_edits") or 0) > 0)],
                [d for d in success if d.get("returned_patch", int(d.get("n_edits") or 0) > 0)],
            )
        else:
            entry["n_failure_patches"] = None
            entry["n_success_patches"] = None
            entry["n_patches"] = None
            entry["patch_files"] = None

        # ── merge ──
        merged = _first(epoch_decisions, "merged_patch")
        entry["merged_patch"] = merged.get("patch") if merged else None
        entry["n_edits_merged"] = merged.get("n_edits") if merged else None

        # ── rank ──
        ranked = _first(epoch_decisions, "ranked_edits")
        entry["ranked_edits"] = ranked.get("edits") if ranked else None
        entry["n_edits_ranked"] = ranked.get("n_edits") if ranked else None

        # ── edit apply ──
        applied = _first(epoch_decisions, "patch_applied") or _first(epoch_decisions, "patch_noop")
        outcomes = None
        if applied:
            report = applied.get("edit_report") or {}
            outcomes = report.get("outcomes")
        entry["edit_apply_report"] = outcomes
        entry["edit_apply_summary"] = (
            _apply_summary(outcomes) if outcomes is not None else None
        )

        # ── val gate ──
        val = (
            _first(epoch_decisions, "val_evaluate")
            or _first(epoch_decisions, "val_cache_hit")
        )
        entry["candidate_hash"] = val.get("skill_hash") if val else None
        entry["selection_hard"] = val.get("score") if val else None

        accept = _first(epoch_decisions, "val_gate_accept")
        if _first(epoch_decisions, "no_edits_skip"):
            entry["action"] = _SKIP_ACTION
        elif _first(epoch_decisions, "patch_refused"):
            # Arm B refused its own edit. A has no such state, so the name is
            # arm B's and the comparison reports it as a divergence.
            entry["action"] = "patch_refused"
        elif accept is not None:
            score = accept.get("cand_score")
            score = float(score) if score is not None else None
            if score is not None and (best is None or score > best):
                entry["action"] = "accept_new_best"
                best = score
            else:
                entry["action"] = "accept"
        elif _first(epoch_decisions, "val_gate_reject"):
            entry["action"] = "reject"
        else:
            entry["action"] = None

        # ── slow update ──
        slow = _first(epoch_decisions, "slow_update")
        skipped = _first(epoch_decisions, "slow_update_skipped")
        if slow is not None:
            # The fields A's slow_result.json also carries, when B recorded them.
            entry["slow_update"] = {
                key: slow[key] for key in _SLOW_UPDATE_FIELDS if key in slow
            }
        elif _first(epoch_decisions, "slow_update_placeholder"):
            # A's first epoch injects empty markers and writes
            # ``{"action": "inject_placeholder"}`` (trainer.py:1824).
            entry["slow_update"] = {"action": "inject_placeholder"}
        elif skipped is not None:
            entry["slow_update"] = {
                "action": "skipped", "reason": skipped.get("reason"),
            }
        else:
            entry["slow_update"] = None

        # ── meta skill ──
        meta = _first(epoch_decisions, "meta_skill")
        entry["meta_skill"] = (
            {key: meta[key] for key in _META_SKILL_FIELDS if key in meta} if meta else None
        )

        # ── early stop ──
        entry["early_stop"] = _first(epoch_decisions, "early_stop") is not None

        history.append(entry)
    return history


# ── runner ─────────────────────────────────────────────────────────────────


def recording_matcher(recording_dir: Path) -> tuple[CallMatcher, dict[str, Any]]:
    """The matcher over A's calls.jsonl, with A's late answers marked.

    A's own timeouts are conceded in replay only: the late answers are never
    served, and the call times out for B as it did for A (CompletionTimeout).
    """
    matcher = CallMatcher(Path(recording_dir) / "calls.jsonl")
    a_timeouts = late_answer_lines(Path(recording_dir) / "out", matcher)
    matcher.mark_late(a_timeouts["lines"])
    return matcher, a_timeouts


def run_replay(
    recording_dir: Path,
    output_dir: Path,
    split_dir: Path | None = None,
    data_path: Path | None = None,
    *,
    claude_backend_wrapper: bool = False,
) -> dict[str, Any]:
    """Run the offline replay: drive arm B's SkillOpt strategy with A's recordings.

    Builds the Loop through ``strategies.create(STRATEGY, MODE_ID, …)`` — the
    same public path ``--strategy skillopt`` takes — so the code under test is
    ``SkillOptLoop`` and the modules it calls, not a second port of A.

    ``split_dir`` is A's id split (35/17/125); its train/val/test ids become the
    run's parts manifest through livemath's ``build_parts_manifest`` (#558).
    ``data_path`` is the LiveMath item corpus ``LiveMathTask`` loads.

    ``claude_backend_wrapper`` runs arm B with that strategy option on (its optimizer
    calls framed as A's Claude backend frames them), and the replay takes the framing
    off before matching, as it does for solver prompts.

    When a ``ReplayMismatch`` occurs (B asks for a call A never made), that
    mismatch is itself the report: it names the call site where B diverges.
    """
    a_out_dir = recording_dir / "out"
    strategy_options = load_a_config(a_out_dir)
    if claude_backend_wrapper:
        strategy_options["claude_backend_wrapper"] = True
    unsupported = unsupported_a_settings(a_out_dir)

    matcher, a_timeouts = recording_matcher(recording_dir)
    client = ReplayClient(matcher, unframe_optimizer=claude_backend_wrapper)
    # Rollout completion order is conceded in replay only — see RecordedRolloutOrder.
    rollout_order = RecordedRolloutOrder(a_out_dir)
    # So is the analysts' completion order — see RecordedPatchOrder.
    patch_order = RecordedPatchOrder(
        matcher, (prompts.MERGE_FAILURE_SYSTEM, prompts.MERGE_SUCCESS_SYSTEM),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "run"

    strategy_class = strategies.resolve(STRATEGY, Path("."))

    report: dict[str, Any] = {
        "recording_dir": str(recording_dir),
        "output_dir": str(output_dir),
        "runs_root": str(runs_root),
        "split_dir": str(split_dir) if split_dir else None,
        "data_path": str(data_path) if data_path else None,
        "mode_id": MODE_ID,
        "strategy": {
            "name": STRATEGY,
            "resolved_by": "package discovery (Package.strategies)",
            "class": f"{strategy_class.__module__}.{strategy_class.__qualname__}",
        },
        "strategy_options": strategy_options,
        "unsupported_a_settings": unsupported,
        "calls_matched": 0,
        "calls_total": matcher.total_recordings,
        "replay_status": "not_started",
        "mismatch": None,
        "unused_recordings": [],
        "rollout_orders": [],
        "patch_orders": [],
        "a_timeouts": a_timeouts["events"],
        "a_timeouts_unresolved": a_timeouts["unresolved"],
        "a_late_answers_after_its_own_timeout": [],
        "a_failed_attempts_replayed": 0,
        "a_events_replayed": [],
        "b_history": [],
        "comparison": None,
    }

    try:
        if unsupported:
            raise ValueError("; ".join(unsupported))

        if split_dir is None:
            split_dir = _resolve_split_dir(a_out_dir)
        if split_dir is None:
            raise FileNotFoundError(
                "no --split-dir given and A's config.json split_dir could not be "
                "resolved; the replay needs A's id split to build the parts manifest"
            )
        split_dir = Path(split_dir)
        report["split_dir"] = str(split_dir)

        # The parts manifest is the only item channel: #558's --parts format,
        # built from A's own split so B evaluates A's 35/17/125.
        parts_path = build_parts_manifest(split_dir, output_dir)

        # Cross-check the split against the size A recorded, so replaying the
        # wrong split fails here instead of diverging in epoch 1 for no clear
        # reason.  A's config names the train size it ran (``train_size: 35``).
        manifest = json.loads(Path(parts_path).read_text())
        a_train_ids = [str(item["id"]) for item in _load_train_items(split_dir)]
        a_val_ids = [str(item["id"]) for item in _load_val_items(split_dir)]
        with open(a_out_dir / "config.json") as f:
            a_train_size = json.load(f).get("train_size")
        if a_train_size and len(a_train_ids) != a_train_size:
            raise ValueError(
                f"{split_dir} has {len(a_train_ids)} train items but A ran with "
                f"train_size={a_train_size}; this is not the split A used"
            )
        report["parts"] = {
            "manifest": str(parts_path),
            "train": len(a_train_ids),
            "val": len(a_val_ids),
            "test": len(manifest.get("test", [])),
        }

        # The initial skill: A's own skill_init file, so epoch 1 starts where A did.
        # It must reach arm B as the seed candidate's `skill` knob (below): every
        # solver and analyst prompt embeds it, so a seed that does not match A's
        # makes every single call mismatch. Refuse an empty one rather than replay
        # a run that cannot match -- the first attempt at this silently compared
        # 0 of 1280 calls because the loaded text was never passed to the loop.
        skill_text = _recorded_seed_skill(recording_dir) or _load_initial_skill(recording_dir)
        report["initial_skill_len"] = len(skill_text)
        if not skill_text.strip():
            raise ValueError(
                "the recording's seed skill could not be loaded (config.json "
                f"skill_init={_skill_init_of(recording_dir)!r}); without it every "
                "prompt differs from A's and the replay compares nothing. Give the "
                "recording a readable skill_init, or its out/skills/skill_v0000.md"
            )

        if data_path is not None:
            os.environ[DATA_PATH_ENV] = str(data_path)
        if not os.environ.get(DATA_PATH_ENV):
            raise FileNotFoundError(
                f"no --data-path given and {DATA_PATH_ENV} is unset; LiveMathTask "
                "needs the item corpus (a directory of qa_*_final.json files)"
            )
        report["data_path"] = os.environ[DATA_PATH_ENV]

        config = LoopConfig(
            epochs=int(strategy_options["max_epochs"]),
            batch_size=1,
            optimizer="random",
            seed=int(strategy_options["seed"]),
            runs_root=runs_root,
            allowed_ops=ALLOWED_OPS,
            strategy=STRATEGY,
            parts_manifest=str(parts_path),
            # Seed arm B with A's own starting skill (loop.py LoopConfig.knobs).
            knobs={SKILL_KNOB: skill_text},
            # Generous: an epoch spends one train rollout, one val gate and two
            # slow-update samples, plus the initial val.
            max_evaluations=4 * int(strategy_options["max_epochs"]) + 16,
            # No held-out audit of the winner: refactory's Loop scores the winner
            # on a holdout after any search (loop.py ``audit_holdout``), which is
            # not part of SkillOpt, so A made no such calls to replay. A's own
            # post-training evaluations (final_selection_eval, test_eval*) are
            # SkillOptLoop's too (``eval_test``) and replay A's recorded calls.
            holdout=0.0,
        )

        loop = strategies.create(
            STRATEGY, MODE_ID, config,
            strategy_options={
                **strategy_options,
                "rollout_order": rollout_order,
                "patch_order": patch_order,
            },
            project=Path("."), model=client,
        )
        loop_report = loop.run()
        report["loop_report"] = _trim_loop_report(loop_report)
        report["test_eval"] = compare_test_eval(
            a_out_dir, (loop_report or {}).get("strategy_report") or {})

        report["replay_status"] = "completed"

    except ReplayMismatch as e:
        report["replay_status"] = "mismatch"
        report["mismatch"] = {
            "request_system": e.request_system[:500],
            "request_prompt": e.request_prompt[:500],
            "nearest_system": e.nearest_system[:500],
            "nearest_prompt": e.nearest_prompt[:500],
            "diff": e.diff_text[:5000],
        }
    except ReplayExhausted as e:
        report["replay_status"] = "exhausted"
        report["mismatch"] = {
            "error": str(e),
            "system": e.system[:500],
            "prompt": e.prompt[:500],
            "consumed_count": e.consumed_count,
        }
    except Exception as e:  # noqa: BLE001 - the report is the product
        report["replay_status"] = "error"
        report["mismatch"] = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # The first divergence is the result, however the run ended. A mismatch inside one
    # item's solver call is isolated by the run (the item scores 0 and the run goes on,
    # #554), so the exception that ends the run -- or a "completed" status -- can come
    # from a later call that only failed because the trajectory had already changed.
    first = matcher.first_divergence
    if first is not None:
        if report["replay_status"] == "error":
            report["error_after_divergence"] = report.get("mismatch")
        if isinstance(first, ReplayExhausted):
            report["replay_status"] = "exhausted"
            report["mismatch"] = {
                "error": str(first), "system": first.system[:500],
                "prompt": first.prompt[:500], "consumed_count": first.consumed_count,
            }
        else:
            report["replay_status"] = "mismatch"
            report["mismatch"] = {
                "request_system": first.request_system[:500],
                "request_prompt": first.request_prompt[:500],
                "nearest_system": first.nearest_system[:500],
                "nearest_prompt": first.nearest_prompt[:500],
                "diff": first.diff_text[:5000],
            }

    if client.harness_error is not None and report["replay_status"] in ("completed", "error"):
        # A fault inside the replay client: whatever B did after it is not a replay.
        report["replay_status"] = "error"
        report["harness_error"] = client.harness_error

    # Whatever happened, report what B did reach and compare it with A.
    report["calls_matched"] = matcher.consumed_count()
    report["unused_after_b_last_served_call"] = matcher.unused_after_last_served()
    report["unused_recordings"] = [
        {"system": system[:80], "prompt": prompt[:80], "remaining": remaining}
        for system, prompt, remaining in matcher.unused_recordings()
    ]
    report["a_late_answers_after_its_own_timeout"] = matcher.late_answers()
    report["a_failed_attempts_replayed"] = matcher.failed_attempts_replayed
    report["a_events_replayed"] = [
        {"call_key": entry["call_key"], "key": entry["key"], "event": entry["a_event"]}
        for entry in client.match_log if "a_event" in entry
    ]
    report["patch_orders"] = [
        {key: value for key, value in entry.items() if key != "order"}
        for entry in patch_order.applied
    ]
    report["rollout_orders"] = [
        {key: value for key, value in entry.items() if key != "order"}
        for entry in rollout_order.applied
    ]

    trace = read_trace(runs_root)
    report["trace_entries"] = len(trace)
    report["strategy_ran"] = bool(trace)
    report["b_history"] = b_history_from_trace(trace)
    if report["b_history"]:
        try:
            report["comparison"] = compare_runs(a_out_dir, report["b_history"])
        except Exception as e:  # noqa: BLE001
            report["comparison_error"] = str(e)
    apply_verdict(report, matcher.unused_before_last_served())

    report_path = output_dir / "replay_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def apply_verdict(report: dict[str, Any], unused_before_last_served: list[dict[str, Any]]) -> None:
    """Decide whether a replay that matched every call also reproduced A's run.

    Matching every call B made is not enough: B can stop early, decide
    differently on data it already had, or skip calls A made. A run stays
    ``completed`` only when the comparison with A's step records finds no
    diverged epoch, B reached every epoch A ran, B's post-training scores equal
    A's summary.json (``compare_test_eval``), and every recording B did not use
    was made after the last one it was served (A's work past the end of B's
    run). Otherwise the status is
    ``diverged`` and ``divergence`` says where, with both sides' values.
    """
    report["unused_before_b_last_served_call"] = len(unused_before_last_served)
    if report.get("replay_status") != "completed":
        return
    comparison = report.get("comparison")
    divergence: dict[str, Any] | None = None
    if not comparison:
        divergence = {
            "phase": "comparison", "field": "comparison",
            "a_value": "A's step records", "b_value": report.get("comparison_error")
            or "no comparison (B recorded no epochs)",
        }
    elif comparison["summary"].get("diverged_epochs") or comparison.get("first_divergence"):
        divergence = comparison.get("first_divergence") or {
            "phase": "comparison", "field": "diverged_epochs",
            "a_value": 0, "b_value": comparison["summary"].get("diverged_epochs"),
        }
    elif (report.get("test_eval") or {}).get("diverged"):
        test_eval = report["test_eval"]
        field = test_eval["diverged"][0]
        divergence = {
            "phase": "test_eval", "field": field,
            "a_value": test_eval["a"].get(field), "b_value": test_eval["b"].get(field),
        }
    elif unused_before_last_served:
        divergence = {
            "phase": "recordings", "field": "unused_before_b_last_served_call",
            "a_value": len(unused_before_last_served), "b_value": 0,
            "examples": unused_before_last_served[:5],
        }
    if divergence is not None:
        report["replay_status"] = "diverged"
        report["divergence"] = divergence


def compare_test_eval(a_out_dir: Path, strategy_report: dict[str, Any]) -> dict[str, Any]:
    """A's post-training scores (summary.json) beside B's (its strategy report).

    Compared on :data:`A_TEST_EVAL_FIELDS` that A recorded; ``diverged`` lists the
    fields whose scores differ (beyond float noise), in A's order.
    """
    summary_path = Path(a_out_dir) / "summary.json"
    a_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    a = {key: a_summary.get(key) for key in A_TEST_EVAL_FIELDS if a_summary.get(key) is not None}
    b = {key: strategy_report.get(key) for key in A_TEST_EVAL_FIELDS}
    diverged = [key for key in a
                if b.get(key) is None or abs(float(a[key]) - float(b[key])) > 1e-9]
    return {"a": a, "b": b, "b_status": strategy_report.get("test_eval"), "diverged": diverged}


def _trim_loop_report(loop_report: Any) -> dict[str, Any]:
    """The few loop-report fields worth keeping beside the comparison."""
    if not isinstance(loop_report, dict):
        return {}
    keep = ("best", "score", "evaluations", "rollouts", "stopped", "reason", "winner")
    return {key: loop_report[key] for key in keep if key in loop_report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the offline SkillOpt replay against arm B"
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        required=True,
        help="Path to A's recording directory",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Path to A's 35/17/125 id split directory (default: A's config split_dir)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=f"LiveMath item corpus (default: ${DATA_PATH_ENV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path for B's output",
    )
    parser.add_argument(
        "--claude-backend-wrapper",
        action="store_true",
        help="run arm B with --strategy-option claude_backend_wrapper=true",
    )
    args = parser.parse_args()

    report = run_replay(
        args.recording_dir, args.output_dir, args.split_dir, args.data_path,
        claude_backend_wrapper=args.claude_backend_wrapper,
    )

    print(f"Replay status: {report['replay_status']}")
    print(f"Strategy: {report['strategy']['class']} ({report['strategy']['resolved_by']})")
    print(f"Arm B strategy ran: {report['strategy_ran']}")
    print(f"Calls total in recording: {report['calls_total']}")
    print(f"Calls matched: {report['calls_matched']}")
    print(f"A's events replayed (timeouts, failed calls): {len(report['a_events_replayed'])}")
    print(f"A's late answers after its own timeout (withheld): "
          f"{len(report['a_late_answers_after_its_own_timeout'])}")

    if report.get("unsupported_a_settings"):
        print("\n=== A SETTINGS ARM B CANNOT MATCH ===")
        for line in report["unsupported_a_settings"]:
            print(f"  - {line}")

    if report.get("test_eval"):
        test_eval = report["test_eval"]
        print("Post-training scores (A | B): " + ", ".join(
            f"{key} {test_eval['a'].get(key)} | {test_eval['b'].get(key)}"
            for key in A_TEST_EVAL_FIELDS))

    if report.get("unused_recordings"):
        n_unused = sum(u["remaining"] for u in report["unused_recordings"])
        print(
            f"Unused recordings: {n_unused} across "
            f"{len(report['unused_recordings'])} keys"
        )

    if report.get("divergence"):
        print("\n=== DIVERGED FROM A ===")
        print(json.dumps(report["divergence"], indent=2, default=str)[:3000])

    if report.get("mismatch"):
        print("\n=== MISMATCH ===")
        print(json.dumps(report["mismatch"], indent=2)[:3000])

    if report.get("comparison"):
        print("\n" + format_report(report["comparison"]))

    sys.exit(0 if report["replay_status"] == "completed" else 1)


if __name__ == "__main__":
    main()
