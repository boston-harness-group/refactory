"""Unit tests for the SkillOpt offline replay infrastructure.

All tests use synthetic recordings — no real benchmark data in the repo.
The tests cover:
  - normalize_recorded_prompt (strips transport wrapping)
  - CallMatcher (content-based matching with positional counters)
  - ReplayClient (executor's agent-node protocol)
  - compare_runs (divergence detection)
  - load_a_config (config mapping)
  - SkillOptTrainer (runs full 6-stage pipeline)
  - Import does not register a strategy
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from refactory_packages.skillopt.replay import (
    CallMatcher,
    ReplayClient,
    ReplayExhausted,
    ReplayMismatch,
    normalize_recorded_prompt,
)
from refactory_packages.skillopt.compare import (
    compare_runs,
    format_report,
    hash_skill,
)
from refactory_packages.skillopt.replay_runner import load_a_config


# ── helpers ────────────────────────────────────────────────────────────────


def _wrap_prompt(content: str) -> str:
    """Apply A's transport wrapping: History:\\n- User: <indented content>\\n\\nAnswer ..."""
    indented = content.replace("\n", "\n  ")
    return f"History:\n- User: {indented}\n\nAnswer the latest user request."


def _make_calls_jsonl(
    tmp_path: Path,
    entries: list[tuple[str, str, str]],
) -> Path:
    """Create a calls.jsonl from (system, prompt_content, result) tuples.

    Wraps prompt_content in A's transport wrapping and result in
    ``{"result": result}`` JSON as stdout.
    """
    path = tmp_path / "calls.jsonl"
    lines = []
    for i, (system, prompt_content, result) in enumerate(entries):
        wrapped_prompt = _wrap_prompt(prompt_content)
        entry = {
            "at": 1000.0 + i,
            "wall_s": 1.0,
            "rc": 0,
            "model": "test-model",
            "effort": "medium",
            "system": system,
            "prompt": wrapped_prompt,
            "stdout": json.dumps({"result": result}),
        }
        lines.append(json.dumps(entry))
    path.write_text("\n".join(lines) + "\n")
    return path


# ── normalizer tests ──────────────────────────────────────────────────────


def test_normalize_strips_wrapper():
    """normalize_recorded_prompt strips the full transport wrapping."""
    content = "What is 2+2?"
    wrapped = _wrap_prompt(content)
    assert normalize_recorded_prompt(wrapped) == content


def test_normalize_strips_wrapper_multiline():
    """Multi-line prompts with 2-space indented continuations."""
    content = "Line one.\nLine two.\nLine three."
    wrapped = _wrap_prompt(content)
    assert normalize_recorded_prompt(wrapped) == content


def test_normalize_strips_structured_tail():
    """Strips ``Return only JSON matching the provided schema.`` tail variant."""
    content = "Give me JSON."
    indented = content.replace("\n", "\n  ")
    wrapped = f"History:\n- User: {indented}\n\nReturn only JSON matching the provided schema."
    assert normalize_recorded_prompt(wrapped) == content


def test_normalize_preserves_inner_history_marker():
    """A prompt whose math text contains literal 'History:\\n- User:' is safe."""
    content = "The History:\n- User: asked about a Radon measure."
    wrapped = _wrap_prompt(content)
    result = normalize_recorded_prompt(wrapped)
    assert result == content


def test_normalize_round_trip():
    """Programmatically applies A's wrapping, then normalises. Round trip."""
    test_cases = [
        "Simple prompt.",
        "History:\n- User: inner marker preserved.",
        "```python\n  x = 1\n  y = 2\n```",
        "",
        "LaTeX: \\(\\mu\\) with {braces}",
        "Line 1\n\nLine 3 (empty line between)",
    ]
    for content in test_cases:
        wrapped = _wrap_prompt(content)
        result = normalize_recorded_prompt(wrapped)
        assert result == content.rstrip(), f"Round-trip failed for: {content!r}"


def test_normalize_preserves_system():
    """System prompts pass through unchanged (stored raw in calls.jsonl)."""
    system = "You are an expert mathematical reasoning agent."
    assert normalize_recorded_prompt(system) == system


# ── matcher tests ─────────────────────────────────────────────────────────


def test_matcher_returns_correct_output(tmp_path):
    """3-entry synthetic calls.jsonl with distinct (system, prompt) pairs."""
    entries = [
        ("sys_a", "prompt_one", "result_one"),
        ("sys_b", "prompt_two", "result_two"),
        ("sys_c", "prompt_three", "result_three"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    assert matcher.lookup("sys_a", "prompt_one") == "result_one"
    assert matcher.lookup("sys_b", "prompt_two") == "result_two"
    assert matcher.lookup("sys_c", "prompt_three") == "result_three"


def test_matcher_handles_duplicates(tmp_path):
    """Same (system, prompt) but different results, ordered by timestamp."""
    entries = [
        ("sys", "same_prompt", "result_first"),
        ("sys", "same_prompt", "result_second"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    assert matcher.lookup("sys", "same_prompt") == "result_first"
    assert matcher.lookup("sys", "same_prompt") == "result_second"


def test_matcher_raises_on_mismatch(tmp_path):
    """lookup() with unrecorded (system, prompt) raises ReplayMismatch."""
    entries = [("sys", "recorded_prompt", "result")]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    with pytest.raises(ReplayMismatch) as exc_info:
        matcher.lookup("sys", "unrecorded_prompt")

    assert exc_info.value.nearest_prompt == "recorded_prompt"
    assert exc_info.value.diff_text  # non-empty diff


def test_matcher_raises_on_exhausted(tmp_path):
    """After consuming all outputs for a key, raises ReplayExhausted."""
    entries = [("sys", "once", "result_once")]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    matcher.lookup("sys", "once")
    with pytest.raises(ReplayExhausted) as exc_info:
        matcher.lookup("sys", "once")
    assert exc_info.value.consumed_count == 1


def test_matcher_reports_unused_per_key(tmp_path):
    """unused_recordings returns keys B never asked for."""
    entries = [
        ("sys_a", "prompt_a", "result_a"),
        ("sys_b", "prompt_b", "result_b"),
        ("sys_c", "prompt_c", "result_c"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    matcher.lookup("sys_a", "prompt_a")
    matcher.lookup("sys_b", "prompt_b")

    unused = matcher.unused_recordings()
    assert len(unused) == 1
    assert unused[0] == ("sys_c", "prompt_c", 1)


def test_matcher_reports_unused_per_recording(tmp_path):
    """unused_recordings tracks per-recording, not per-key.

    If A has 3 identical calls and B consumes only 1, reports 2 remaining.
    """
    entries = [
        ("sys", "dup", "r1"),
        ("sys", "dup", "r2"),
        ("sys", "dup", "r3"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    matcher.lookup("sys", "dup")  # consumes r1

    unused = matcher.unused_recordings()
    assert len(unused) == 1
    assert unused[0] == ("sys", "dup", 2)


def test_matcher_total_recordings(tmp_path):
    """total_recordings counts individual recordings."""
    entries = [
        ("sys", "a", "r1"),
        ("sys", "a", "r2"),
        ("sys", "b", "r3"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)
    assert matcher.total_recordings == 3


def test_matcher_consumed_count(tmp_path):
    """consumed_count tracks how many recordings have been served."""
    entries = [
        ("sys", "a", "r1"),
        ("sys", "a", "r2"),
        ("sys", "b", "r3"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)

    assert matcher.consumed_count() == 0
    matcher.lookup("sys", "a")
    assert matcher.consumed_count() == 1
    matcher.lookup("sys", "b")
    assert matcher.consumed_count() == 2


# ── client tests ──────────────────────────────────────────────────────────


def _make_client(tmp_path: Path) -> ReplayClient:
    entries = [("test_sys", "test_prompt", "test_result")]
    path = _make_calls_jsonl(tmp_path, entries)
    matcher = CallMatcher(path)
    return ReplayClient(matcher)


def test_client_complete_returns_text(tmp_path):
    """complete() returns the matched text."""
    client = _make_client(tmp_path)
    result = client.complete("test_prompt", system="test_sys")
    assert result == "test_result"


def test_client_accepts_call_key(tmp_path):
    """complete() with call_key works without error."""
    client = _make_client(tmp_path)
    result = client.complete(
        "test_prompt",
        system="test_sys",
        call_key=("analyst_failure", 1, 0, 0),
    )
    assert result == "test_result"


def test_client_accepts_workdir(tmp_path):
    """complete() with workdir works without error."""
    client = _make_client(tmp_path)
    result = client.complete(
        "test_prompt", system="test_sys", workdir="/tmp"
    )
    assert result == "test_result"


def test_client_for_node(tmp_path):
    """for_node() returns self."""
    client = _make_client(tmp_path)
    assert client.for_node("solve", role="solver") is client


def test_client_with_tool_policy(tmp_path):
    """with_tool_policy() returns self."""
    client = _make_client(tmp_path)
    assert client.with_tool_policy("none") is client


def test_client_with_wall(tmp_path):
    """with_wall() returns self."""
    client = _make_client(tmp_path)
    assert client.with_wall(300) is client


def test_client_route(tmp_path):
    """route() returns {'worker_backend': 'dsh'}."""
    client = _make_client(tmp_path)
    assert client.route() == {"worker_backend": "dsh"}


def test_client_match_log(tmp_path):
    """After complete(), match_log has entries with key, index, call_key."""
    entries = [
        ("sys1", "p1", "r1"),
        ("sys2", "p2", "r2"),
    ]
    path = _make_calls_jsonl(tmp_path, entries)
    client = ReplayClient(CallMatcher(path))

    client.complete("p1", system="sys1", call_key="ck1")
    client.complete("p2", system="sys2", call_key="ck2")

    assert len(client.match_log) == 2
    assert client.match_log[0]["call_key"] == "ck1"
    assert client.match_log[0]["index"] == 0
    assert client.match_log[1]["call_key"] == "ck2"


# ── comparison tests ──────────────────────────────────────────────────────


def _make_a_out(tmp_path: Path, epochs: list[dict], *,
                config: dict | None = None,
                skills: dict[int, str] | None = None) -> Path:
    """Create a synthetic A out/ directory."""
    a_out = tmp_path / "a_out"
    a_out.mkdir()
    (a_out / "history.json").write_text(json.dumps(epochs))
    (a_out / "skills").mkdir()
    (a_out / "meta_skill").mkdir()
    (a_out / "slow_update").mkdir()
    (a_out / "steps").mkdir()
    if config:
        (a_out / "config.json").write_text(json.dumps(config))
    if skills:
        for version, text in skills.items():
            (a_out / "skills" / f"skill_v{version:04d}.md").write_text(text)
    return a_out


def test_comparison_detects_divergence(tmp_path):
    """Planted divergence at epoch 2 is found by compare_runs."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2,
            "rollout_hard": 0.6, "rollout_n": 35,
            "n_patches": 3, "n_failure_patches": 1,
            "n_success_patches": 2, "n_edits_merged": 5, "n_edits_ranked": 3,
            "edit_apply_summary": {"total": 3, "applied": 3, "skipped": 0, "errors": 0},
            "candidate_hash": "ef567890ef567890",
            "selection_hard": 0.6, "action": "accept_new_best",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    # B matches epoch 1, diverges at epoch 2 (different val score)
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2,
            "rollout_hard": 0.6, "rollout_n": 35,
            "n_patches": 3, "n_failure_patches": 1,
            "n_success_patches": 2, "n_edits_merged": 5, "n_edits_ranked": 3,
            "edit_apply_summary": {"total": 3, "applied": 3, "skipped": 0, "errors": 0},
            "candidate_hash": "ef567890ef567890",
            "selection_hard": 0.7,  # ← divergence: 0.7 vs A's 0.6
            "action": "accept_new_best",
        },
    ]

    report = compare_runs(a_out, b_history)

    assert report["first_divergence"] is not None
    assert report["first_divergence"]["epoch"] == 2
    assert report["first_divergence"]["field"] == "val_scores"
    assert report["first_divergence"]["a_value"] == 0.6
    assert report["first_divergence"]["b_value"] == 0.7


def test_comparison_reports_match(tmp_path):
    """Identical A and B outputs → first_divergence=None."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]

    report = compare_runs(a_out, b_history)
    assert report["first_divergence"] is None


def test_comparison_report_covers_all_fields(tmp_path):
    """Every per-epoch entry has all brief-required fields."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.4, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.4, "action": "reject",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.4, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.4, "action": "reject",
        },
    ]

    report = compare_runs(a_out, b_history)
    required_fields = [
        "rollout_hard", "reflection_groups", "merged_edits", "ranked_edits",
        "edit_apply_outcomes", "candidate_skill_hash", "val_scores",
        "accept_reject", "slow_update", "meta_skill", "early_stop",
    ]
    for ep in report["per_epoch"]:
        for field in required_fields:
            assert field in ep, f"Missing field {field} in epoch {ep.get('epoch')}"


def test_comparison_detects_rollout_hard_divergence(tmp_path):
    """When A and B have different rollout_hard, the comparison catches it."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.8, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.3, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.5, "action": "accept",
        },
    ]

    report = compare_runs(a_out, b_history)
    assert report["first_divergence"] is not None
    assert report["first_divergence"]["field"] == "rollout_hard"
    assert report["first_divergence"]["a_value"] == 0.8
    assert report["first_divergence"]["b_value"] == 0.3


def test_comparison_detects_missing_b_fields(tmp_path):
    """When B lacks fields A has, the comparison is not vacuously empty."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept_new_best",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    # B only has rollout_hard and rollout_n — missing all other fields
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
        },
    ]

    report = compare_runs(a_out, b_history)
    # Should NOT report as fully matched since B is missing most fields
    assert report["summary"]["matched_epochs"] == 0
    assert report["summary"]["diverged_epochs"] == 1


def test_comparison_detects_missing_b_epoch(tmp_path):
    """When B doesn't reach an epoch A completed, first_divergence reports it."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.5, "action": "accept",
        },
        {
            "epoch": 2, "step": 2,
            "rollout_hard": 0.6, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "2222333344445555",
            "selection_hard": 0.6, "action": "accept_new_best",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.5, "action": "accept",
        },
    ]

    report = compare_runs(a_out, b_history)
    assert report["first_divergence"] is not None
    assert report["first_divergence"]["epoch"] == 2
    assert report["first_divergence"]["field"] == "epoch_reached"


def test_comparison_detects_early_stop_from_data(tmp_path):
    """A's early stop is derived from config vs history length, not hardcoded."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.8, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2,
            "rollout_hard": 0.6, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "2222333344445555",
            "selection_hard": 0.7, "action": "reject",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs, config={"num_epochs": 3})

    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.8, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2,
            "rollout_hard": 0.6, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "2222333344445555",
            "selection_hard": 0.7, "action": "reject",
            "early_stop": True,
        },
    ]

    report = compare_runs(a_out, b_history)

    last_epoch = report["per_epoch"][-1]
    assert last_epoch["early_stop"]["a"] is True
    assert last_epoch["early_stop"]["b"] is True
    assert last_epoch["early_stop"]["match"] is True


def test_comparison_uses_skill_files(tmp_path):
    """Candidate skill hash comparison uses A's actual skill files."""
    skill_text = "# My Skill v1\nSome content here."
    expected_hash = hash_skill(skill_text)

    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.4, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": expected_hash,
            "selection_hard": 0.4, "action": "reject",
        },
    ]
    a_out = _make_a_out(
        tmp_path, a_epochs,
        skills={1: skill_text},
    )

    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.4, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": expected_hash,
            "selection_hard": 0.4, "action": "reject",
        },
    ]

    report = compare_runs(a_out, b_history)
    ep = report["per_epoch"][0]

    assert ep["candidate_skill_hash"]["a"] == expected_hash
    assert ep["candidate_skill_hash"]["a_computed_from_file"] == expected_hash
    assert ep["candidate_skill_hash"]["a_consistent"] is True
    assert ep["candidate_skill_hash"]["match"] is True


# ── hash_skill ────────────────────────────────────────────────────────────


def test_hash_skill():
    """hash_skill() produces a 16-char hex string."""
    text = "# Live Math Skill v1\nSome content here."
    h = hash_skill(text)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
    expected = hashlib.sha256(text.encode()).hexdigest()[:16]
    assert h == expected


# ── config mapping ────────────────────────────────────────────────────────


def test_config_mapping(tmp_path):
    """load_a_config maps all 9 critical parameters."""
    a_config = {
        "num_epochs": 10,
        "early_stop_patience": 3,
        "early_stop_at_full_val": True,
        "edit_budget": 4,
        "min_edit_budget": 2,
        "seed": 42,
        "slow_update_samples": 20,
        "minibatch_size": 8,
        "merge_batch_size": 8,
        # Extra keys that should be ignored
        "model_backend": "claude_chat",
        "env": "livemathematicianbench",
    }
    a_out = tmp_path / "a_out"
    a_out.mkdir()
    (a_out / "config.json").write_text(json.dumps(a_config))

    mapped = load_a_config(a_out)

    assert mapped["max_epochs"] == 10
    assert mapped["early_stop_patience"] == 3
    assert mapped["early_stop_at_full_val"] is True
    assert mapped["edit_budget_max"] == 4
    assert mapped["edit_budget_min"] == 2
    assert mapped["seed"] == 42
    assert mapped["slow_update_samples"] == 20
    assert mapped["group_size"] == 8
    assert mapped["merge_batch_size"] == 8
    assert len(mapped) == 9  # exactly 9 keys


# ── trainer produces complete history entries ─────────────────────────────


def test_trainer_produces_all_fields(tmp_path):
    """SkillOptTrainer produces B history entries with all required fields.

    Creates a synthetic recording with solver + analyst + merge + rank calls,
    then runs the trainer through the full pipeline. The resulting B history
    entry must contain all fields the comparison requires.
    """
    from refactory_packages.skillopt.trainer import (
        SkillOptTrainer,
        build_solver_system,
        build_solver_user,
    )
    from refactory_packages.livemath.shuffling import shuffle_item_choices

    item = {
        "id": "test:1",
        "question": "What is 2+2?",
        "choices": [
            {"label": "A", "text": "3"},
            {"label": "B", "text": "4"},
            {"label": "C", "text": "5"},
        ],
        "correct_choice": {"label": "B", "text": "4"},
    }
    val_item = {
        "id": "val:1",
        "question": "What is 1+1?",
        "choices": [
            {"label": "A", "text": "1"},
            {"label": "B", "text": "2"},
            {"label": "C", "text": "3"},
        ],
        "correct_choice": {"label": "B", "text": "2"},
    }

    # Build expected prompts
    skill_text = ""
    system = build_solver_system(skill_text)
    batch_seed = 42 + 1 * 1000 + 0 + 1  # compute_batch_seed(42, 1)
    shuffled = shuffle_item_choices(item, batch_seed)
    user = build_solver_user(shuffled)

    val_shuffled = shuffle_item_choices(val_item, 42)
    val_user = build_solver_user(val_shuffled)

    # Create calls.jsonl with all needed calls (wrapped in transport format)
    analyst_response = json.dumps({
        "batch_size": 1,
        "failure_summary": [{"failure_type": "other", "count": 1, "description": "wrong"}],
        "patch": {
            "reasoning": "fix it",
            "edits": [{"op": "append", "content": "## Fix\nCheck carefully."}],
        },
    })

    calls = [
        # Solver rollout call
        {
            "at": 1000.0, "wall_s": 1.0, "rc": 0, "model": "test",
            "effort": "medium", "system": system,
            "prompt": _wrap_prompt(user),
            "stdout": json.dumps({"result": "I think the answer is <answer>A</answer>"}),
        },
        # Analyst failure call (will be matched by content)
        {
            "at": 1001.0, "wall_s": 1.0, "rc": 0, "model": "test",
            "effort": "medium",
            "system": "placeholder_analyst_system",
            "prompt": _wrap_prompt("placeholder_analyst_user"),
            "stdout": json.dumps({"result": analyst_response}),
        },
        # Val eval call
        {
            "at": 1002.0, "wall_s": 1.0, "rc": 0, "model": "test",
            "effort": "medium", "system": system,
            "prompt": _wrap_prompt(val_user),
            "stdout": json.dumps({"result": "<answer>B</answer>"}),
        },
    ]

    # For this test, we use a direct model stub instead of CallMatcher
    # to ensure the trainer produces all fields.
    call_idx = [0]
    responses = {
        "solver_train": "I think the answer is <answer>A</answer>",
        "analyst": analyst_response,
        "solver_val": "<answer>B</answer>",
    }

    class StubModel:
        def __init__(self):
            self.calls = []

        def complete(self, prompt, *, system="", **kw):
            self.calls.append((system[:50], prompt[:50]))
            # Return appropriate response based on system prompt
            if "failure-analysis" in system or "success-pattern" in system:
                return analyst_response
            if "skill-edit coordinator" in system or "FINAL merge" in system:
                return json.dumps({
                    "reasoning": "merged",
                    "edits": [{"op": "append", "content": "## Merged\nContent"}],
                })
            if "skill-optimization" in system:
                return json.dumps({
                    "reasoning": "ranked",
                    "selected_indices": [0],
                })
            # Solver calls
            if "mathematical reasoning" in system:
                return "<answer>A</answer>"
            return "<answer>X</answer>"

    model = StubModel()

    config = {
        "max_epochs": 1,
        "early_stop_patience": 0,
        "early_stop_at_full_val": False,
        "edit_budget_max": 2,
        "edit_budget_min": 1,
        "seed": 42,
        "slow_update_samples": 5,
        "group_size": 4,
        "merge_batch_size": 4,
        "use_slow_update": False,
        "use_meta_skill": False,
    }

    output_dir = tmp_path / "output"
    trainer = SkillOptTrainer(
        model=model,
        config=config,
        train_items=[item],
        val_items=[val_item],
        initial_skill="",
        output_dir=output_dir,
    )

    history = trainer.run()
    assert len(history) == 1

    entry = history[0]
    # Check all required fields are present
    assert "epoch" in entry
    assert "step" in entry
    assert "rollout_hard" in entry
    assert "rollout_n" in entry
    assert "n_patches" in entry
    assert "n_failure_patches" in entry
    assert "n_success_patches" in entry
    assert "n_edits_merged" in entry
    assert "n_edits_ranked" in entry
    assert "candidate_hash" in entry
    assert "selection_hard" in entry
    assert "action" in entry

    # The trainer used the stub model, not the replay client
    assert len(model.calls) > 0  # Made multiple calls


# ── trainer uses livemath prompt construction ─────────────────────────────


def test_trainer_uses_livemath_prompts(tmp_path):
    """The trainer builds solver prompts through livemath's code, not reimplemented helpers.

    Verifies that the system and user prompts match what livemath/mode.py
    and livemath/ops.py produce.
    """
    from refactory_packages.skillopt.trainer import build_solver_system, build_solver_user
    from refactory_packages.livemath.mode import SYSTEM_PROMPT
    from refactory_packages.livemath.ops import format_prompt

    # Check system prompt matches livemath template
    skill = "# Test Skill\nSome rules."
    system = build_solver_system(skill)
    expected_section = f"## Skill\n{skill.strip()}\n\n"
    expected = SYSTEM_PROMPT.replace("{{skill_section}}", expected_section).rstrip()
    assert system == expected

    # Check user prompt matches livemath ops
    item = {
        "question": "What is 2+2?",
        "choices": [
            {"label": "A", "text": "3"},
            {"label": "B", "text": "4"},
        ],
    }
    user = build_solver_user(item)
    assert "## Question\nWhat is 2+2?" in user
    assert "A. 3" in user
    assert "B. 4" in user


# ── import does not register ──────────────────────────────────────────────


def test_replay_import_does_not_register():
    """Importing replay.py does NOT add 'skillopt' to strategies.STRATEGIES."""
    from refactory_dsh import strategies

    before = set(strategies.STRATEGIES.keys())

    import importlib
    import refactory_packages.skillopt.replay
    importlib.reload(refactory_packages.skillopt.replay)

    after = set(strategies.STRATEGIES.keys())
    assert "skillopt" not in after, (
        "importing replay.py registered 'skillopt' as a strategy"
    )
    assert before == after, (
        f"importing replay.py changed STRATEGIES: added {after - before}"
    )


# ── format_report ─────────────────────────────────────────────────────────


def test_format_report_includes_divergence(tmp_path):
    """format_report renders the first divergence info."""
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.8,  # divergence
            "action": "accept",
        },
    ]

    report = compare_runs(a_out, b_history)
    text = format_report(report)

    assert "First Divergence" in text
    assert "val_scores" in text
    assert "Epoch" in text


def test_trainer_edits_have_source_type_and_support_count(tmp_path):
    """B's trainer adds source_type and support_count to each edit dict.

    Mirrors A's _normalise_patches (skillopt/engine/trainer.py:157-186, MIT)
    which calls item.setdefault('source_type', ...) and
    item.setdefault('support_count', ...) on each edit.
    """
    from refactory_packages.skillopt.trainer import SkillOptTrainer

    analyst_response = json.dumps({
        "batch_size": 5,
        "patch": {
            "reasoning": "test",
            "edits": [
                {"op": "append", "content": "## Edit 1"},
                {"op": "append", "content": "## Edit 2"},
            ],
        },
    })

    class StubModel:
        def complete(self, prompt, *, system="", **kw):
            if "failure-analysis" in system or "success-pattern" in system:
                return analyst_response
            if "skill-edit coordinator" in system or "FINAL merge" in system:
                return json.dumps({
                    "reasoning": "merged",
                    "edits": [{"op": "append", "content": "## Merged"}],
                })
            if "skill-optimization" in system:
                return json.dumps({
                    "reasoning": "ranked",
                    "selected_indices": [0],
                })
            return "<answer>A</answer>"

    item = {
        "id": "test:1",
        "question": "What is 2+2?",
        "choices": [
            {"label": "A", "text": "3"},
            {"label": "B", "text": "4"},
        ],
        "correct_choice": {"label": "B", "text": "4"},
    }
    val_item = {
        "id": "val:1",
        "question": "What is 1+1?",
        "choices": [
            {"label": "A", "text": "1"},
            {"label": "B", "text": "2"},
        ],
        "correct_choice": {"label": "B", "text": "2"},
    }

    config = {
        "max_epochs": 1,
        "early_stop_patience": 0,
        "early_stop_at_full_val": False,
        "edit_budget_max": 4,
        "edit_budget_min": 1,
        "seed": 42,
        "slow_update_samples": 5,
        "group_size": 4,
        "merge_batch_size": 4,
        "use_slow_update": False,
        "use_meta_skill": False,
    }

    output_dir = tmp_path / "output"
    trainer = SkillOptTrainer(
        model=StubModel(),
        config=config,
        train_items=[item],
        val_items=[val_item],
        initial_skill="",
        output_dir=output_dir,
    )

    # Run the reflect stage directly to inspect patch edits
    failure_patches, success_patches, all_raw = trainer._reflect(
        failures=[item],
        successes=[],
        conversations={"test:1": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
        skill_text="",
        edit_budget=4,
        step_dir=output_dir,
    )

    assert len(failure_patches) > 0
    for patch in failure_patches:
        for edit in patch.get("edits", []):
            assert "source_type" in edit, (
                "edit missing source_type — porting bug from "
                "skillopt/engine/trainer.py:157-186"
            )
            assert "support_count" in edit, (
                "edit missing support_count — porting bug from "
                "skillopt/engine/trainer.py:157-186"
            )
            assert edit["source_type"] == "failure"
            assert edit["support_count"] == 5  # from batch_size=5


def test_slow_update_boolean_vs_dict_match(tmp_path):
    """slow_update comparison handles B storing True vs A storing a dict.

    B may record slow_update=True while A has the full result dict.
    The comparison should treat this as a presence match, not a divergence.
    """
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    # Create A's slow_update result as a full dict
    slow_dir = a_out / "slow_update" / "epoch_1"
    slow_dir.mkdir(parents=True)
    (slow_dir / "slow_result.json").write_text(json.dumps({
        "updated": True, "accuracy": 0.6, "details": {"n": 20},
    }))

    # B stores just True
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
            "slow_update": True,
        },
    ]

    report = compare_runs(a_out, b_history)
    ep = report["per_epoch"][0]
    assert ep["slow_update"]["match"] is True, (
        "slow_update should match when B has True and A has a dict"
    )


def test_meta_skill_boolean_vs_dict_match(tmp_path):
    """meta_skill comparison handles B storing True vs A storing a dict.

    B may record meta_skill=True while A has the full result dict.
    The comparison should treat this as a presence match, not a divergence.
    """
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    # Create A's meta_skill result as a full dict
    meta_dir = a_out / "meta_skill" / "epoch_1"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta_skill_result.json").write_text(json.dumps({
        "notes": ["Improve step-by-step reasoning"],
    }))

    # B stores just True
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
            "meta_skill": True,
        },
    ]

    report = compare_runs(a_out, b_history)
    ep = report["per_epoch"][0]
    assert ep["meta_skill"]["match"] is True, (
        "meta_skill should match when B has True and A has a dict"
    )


def test_format_report_no_false_match_claim(tmp_path):
    """format_report does not claim 'match completely' when epochs are diverged.

    When first_divergence is None but _epoch_matched marks epochs as
    diverged (e.g. B missing fields A has), the summary must not say
    'A and B match completely'.
    """
    a_epochs = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5, "rollout_n": 35,
            "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    # B is missing most fields → diverged_epochs > 0 but first_divergence
    # may be None (no explicit field match=False, just None+A-has-data).
    b_history = [
        {
            "epoch": 1, "step": 1,
            "rollout_hard": 0.5,
        },
    ]

    report = compare_runs(a_out, b_history)
    text = format_report(report)

    # Should NOT say "match completely" when diverged_epochs > 0
    assert "match completely" not in text, (
        "format_report says 'match completely' but diverged_epochs > 0"
    )
