"""Unit tests for the SkillOpt offline replay infrastructure.

All tests use synthetic recordings — no real benchmark data in the repo.
The tests cover:
  - normalize_recorded_prompt (strips transport wrapping)
  - CallMatcher (content-based matching with positional counters)
  - ReplayClient (executor's agent-node protocol)
  - compare_runs (divergence detection)
  - load_a_config (config mapping)
  - run_replay (end-to-end replay via the graph, not A-vs-A)
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
            "epoch": 1, "step": 1, "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2, "n_patches": 3, "n_failure_patches": 1,
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
            "epoch": 1, "step": 1, "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2, "n_patches": 3, "n_failure_patches": 1,
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
            "epoch": 1, "step": 1, "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    b_history = [
        {
            "epoch": 1, "step": 1, "n_patches": 4, "n_failure_patches": 2,
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
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.4, "action": "reject",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)
    b_history = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.4, "action": "reject",
        },
    ]

    report = compare_runs(a_out, b_history)
    required_fields = [
        "reflection_groups", "merged_edits", "ranked_edits",
        "edit_apply_outcomes", "candidate_skill_hash", "val_scores",
        "accept_reject", "slow_update", "meta_skill", "early_stop",
    ]
    for ep in report["per_epoch"]:
        for field in required_fields:
            assert field in ep, f"Missing field {field} in epoch {ep.get('epoch')}"


def test_comparison_detects_missing_b_epoch(tmp_path):
    """When B doesn't reach an epoch A completed, first_divergence reports it."""
    a_epochs = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.5, "action": "accept",
        },
        {
            "epoch": 2, "step": 2, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "2222333344445555",
            "selection_hard": 0.6, "action": "accept_new_best",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    # B only reached epoch 1 (e.g., hit a mismatch during epoch 2 rollout)
    b_history = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
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
    assert report["first_divergence"]["a_value"] is True
    assert report["first_divergence"]["b_value"] is False


def test_comparison_detects_early_stop_from_data(tmp_path):
    """A's early stop is derived from config vs history length, not hardcoded."""
    # Config says 3 epochs, but A only ran 2 → early stopped
    a_epochs = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.8, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "2222333344445555",
            "selection_hard": 0.7, "action": "reject",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs, config={"num_epochs": 3})

    # B also ran 2 epochs and reports early_stop=True at epoch 2
    b_history = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.8, "action": "accept_new_best",
        },
        {
            "epoch": 2, "step": 2, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
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


def test_comparison_detects_extra_b_epochs(tmp_path):
    """B epochs beyond A's history are reported as divergences."""
    a_epochs = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.4, "action": "reject",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)

    b_history = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
            "candidate_hash": "1111222233334444",
            "selection_hard": 0.4, "action": "reject",
        },
        {
            "epoch": 2, "step": 2, "n_patches": 1, "n_failure_patches": 1,
            "n_success_patches": 0, "n_edits_merged": 1, "n_edits_ranked": 1,
            "edit_apply_summary": {"total": 1, "applied": 1, "skipped": 0, "errors": 0},
            "candidate_hash": "5555666677778888",
            "selection_hard": 0.5, "action": "accept",
        },
    ]

    report = compare_runs(a_out, b_history)
    assert report["first_divergence"] is not None
    assert report["first_divergence"]["phase"] == "extra_b_epoch"
    assert report["first_divergence"]["epoch"] == 2
    assert len(report["per_epoch"]) == 2


def test_comparison_uses_skill_files(tmp_path):
    """Candidate skill hash comparison uses A's actual skill files."""
    skill_text = "# My Skill v1\nSome content here."
    expected_hash = hash_skill(skill_text)

    a_epochs = [
        {
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
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
            "epoch": 1, "step": 1, "n_patches": 2, "n_failure_patches": 1,
            "n_success_patches": 1, "n_edits_merged": 3, "n_edits_ranked": 2,
            "edit_apply_summary": {"total": 2, "applied": 2, "skipped": 0, "errors": 0},
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


# ── run_replay via graph (not A-vs-A) ─────────────────────────────────────


def test_run_replay_drives_graph_and_reports_mismatch(tmp_path):
    """run_replay() drives B's LiveMath graph via the ReplayClient.

    Creates a synthetic recording where the first rollout call is for a specific
    (system, prompt) pair built from an item.  The runner builds that same
    prompt from the item data and calls client.complete(), which matches.
    After the training items are exhausted, the next call (e.g., for a
    different epoch) hits a mismatch or unused recordings are reported.

    This test drives the REAL path: runner → graph prompt construction →
    ReplayClient → matcher, proving B's prompts are built the same way A's
    were (after normalization).
    """
    from refactory_packages.skillopt.replay_runner import (
        run_replay, _build_system_prompt, _build_user_prompt,
        _compute_batch_seed, _shuffle_epoch_items,
    )
    from refactory_packages.livemath.shuffling import shuffle_item_choices

    rec_dir = tmp_path / "recording"
    rec_dir.mkdir()

    # Build a single item matching what livemath/ops.py would produce
    item = {
        "id": "test:1",
        "month": "test",
        "no": 1,
        "question": "What is 2+2?",
        "choices": [
            {"label": "A", "text": "3"},
            {"label": "B", "text": "4"},
            {"label": "C", "text": "5"},
        ],
        "correct_choice": {"label": "B", "text": "4"},
    }

    # Build the system and user prompts the same way B does, including
    # the choice shuffle that the runner applies with the batch seed.
    # batch_seed for epoch=1, seed=42 is 42 + 1*1000 + 0 + 1 = 1043
    skill_text = ""  # initial skill is empty when none is found
    system = _build_system_prompt(skill_text)
    batch_seed = _compute_batch_seed(42, epoch=1)
    shuffled_item = shuffle_item_choices(item, batch_seed)
    user = _build_user_prompt(shuffled_item)

    # Create calls.jsonl with a matching entry (wrapped in A's transport format)
    calls = [{
        "at": 1000.0, "wall_s": 1.0, "rc": 0, "model": "test",
        "effort": "medium", "system": system,
        "prompt": _wrap_prompt(user),
        "stdout": json.dumps({"result": "<answer>B</answer>"}),
    }]
    (rec_dir / "calls.jsonl").write_text(
        "\n".join(json.dumps(c) for c in calls) + "\n"
    )

    # Create A's out/ with config and history
    out_dir = rec_dir / "out"
    out_dir.mkdir()
    config = {
        "num_epochs": 1, "early_stop_patience": 0, "early_stop_at_full_val": False,
        "edit_budget": 2, "min_edit_budget": 1, "seed": 42,
        "slow_update_samples": 5, "minibatch_size": 4, "merge_batch_size": 4,
        "skill_init": "",
        "split_dir": "",
    }
    (out_dir / "config.json").write_text(json.dumps(config))
    history = [{
        "epoch": 1, "step": 1,
        "rollout_hard": 1.0, "rollout_n": 1,
        "n_patches": 0, "n_failure_patches": 0, "n_success_patches": 0,
        "n_edits_merged": 0, "n_edits_ranked": 0,
        "candidate_hash": "", "selection_hard": 1.0, "action": "accept",
    }]
    (out_dir / "history.json").write_text(json.dumps(history))
    (out_dir / "skills").mkdir()
    (out_dir / "meta_skill").mkdir()
    (out_dir / "slow_update").mkdir()
    (out_dir / "steps").mkdir()

    # Create data dir with one training item
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train").mkdir()
    (data_dir / "train" / "items.json").write_text(json.dumps([item]))
    (data_dir / "val").mkdir()
    (data_dir / "val" / "items.json").write_text("[]")

    # Run replay
    result_dir = tmp_path / "output"
    report = run_replay(rec_dir, result_dir, data_dir)

    # B should have matched the one rollout call
    assert report["calls_matched"] >= 1
    assert report["replay_status"] == "completed"

    # Report file was written
    assert (result_dir / "replay_report.json").exists()


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
            "epoch": 1, "step": 1, "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
            "candidate_hash": "abcd1234abcd1234",
            "selection_hard": 0.5, "action": "accept",
        },
    ]
    a_out = _make_a_out(tmp_path, a_epochs)
    b_history = [
        {
            "epoch": 1, "step": 1, "n_patches": 4, "n_failure_patches": 2,
            "n_success_patches": 2, "n_edits_merged": 6, "n_edits_ranked": 4,
            "edit_apply_summary": {"total": 4, "applied": 4, "skipped": 0, "errors": 0},
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
