"""Unit tests for the SkillOpt offline replay infrastructure.

All tests use synthetic recordings — no real benchmark data in the repo.
The tests cover:
  - normalize_recorded_prompt (strips transport wrapping)
  - CallMatcher (content-based matching with positional counters)
  - ReplayClient (executor's agent-node protocol)
  - RecordedPatchOrder / RecordedRolloutOrder (replay-only orders, set at the source)
  - compare_runs (divergence detection)
  - load_a_config (config mapping)
  - b_history_from_trace (arm B's trace → the comparison's B side)
  - run_replay (the replay drives arm B's SkillOptLoop end to end)
  - Import does not register a strategy
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from refactory_packages.skillopt.trace import current_call_key, optimizer_call
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


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """``trace.ask`` waits 1, 2, 4 s after failed attempts, as SkillOpt does; the
    suite records the waits instead of sleeping them."""
    waits: list[float] = []
    monkeypatch.setattr("refactory_packages.skillopt.trace._sleep", waits.append)
    return waits


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


def test_nearest_prefers_the_same_system(tmp_path):
    """The divergence report's nearest recording is searched under the
    request's own system first (a line-level ratio, so 130 KB prompts stay fast)."""
    calls = _make_calls_jsonl(tmp_path, [
        ("sys_a", "line one\nline two\nline three", "a"),
        ("sys_b", "line one\nline two\nline 3", "b"),
    ])
    matcher = CallMatcher(calls)
    with pytest.raises(ReplayMismatch) as exc_info:
        matcher.lookup("sys_b", "line one\nline two\nline three\nline four")
    assert exc_info.value.nearest_system == "sys_b"
    other = CallMatcher(calls)
    with pytest.raises(ReplayMismatch) as exc_info:
        other.lookup("sys_c", "line one\nline two\nline three\nline four")
    assert exc_info.value.nearest_prompt == "line one\nline two\nline three"
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


def test_a_trailing_newline_on_the_system_prompt_still_matches(tmp_path):
    """A's recorded system prompts lost the trailing newline SkillOpt's _build_system keeps.

    Arm B reproduces SkillOpt's bytes, so the same system prompt arrives with one more
    newline than the recording; on the first real s42 replay that one byte mismatched
    every solver call. Trailing whitespace is transport, not content.
    """
    path = _make_calls_jsonl(tmp_path, [("You are a solver.\n\n<answer>B</answer>", "q", "r")])
    matcher = CallMatcher(path)
    assert matcher.lookup("You are a solver.\n\n<answer>B</answer>\n", "q\n") == "r"
    assert matcher.consumed_for("You are a solver.\n\n<answer>B</answer>\n", "q") == 1


def test_a_system_only_difference_shows_in_the_diff(tmp_path):
    """When only the system prompt differs, the prompt diff is empty; show the system's."""
    path = _make_calls_jsonl(tmp_path, [("system A", "same prompt", "r")])
    matcher = CallMatcher(path)
    with pytest.raises(ReplayMismatch) as caught:
        matcher.lookup("system B", "same prompt")
    assert "-system B" in caught.value.diff_text and "+system A" in caught.value.diff_text


def test_the_first_divergence_is_kept_and_ends_the_replay(tmp_path):
    """The first mismatch is the result, even when its caller swallows it.

    A solver call that mismatches fails one item, which the run isolates (the item
    scores 0 and the run goes on), so a later call -- reached only because the
    trajectory changed -- would otherwise be reported as the divergence. Every call
    after the first divergence raises at once, matched or not.
    """
    path = _make_calls_jsonl(tmp_path, [("sys", "first", "r1"), ("sys", "second", "r2")])
    matcher = CallMatcher(path)
    with pytest.raises(ReplayMismatch):
        matcher.lookup("sys", "unrecorded")          # swallowed by an isolated item
    with pytest.raises(ReplayMismatch):
        matcher.lookup("sys", "second")              # recorded, but after the divergence
    assert matcher.first_divergence is not None
    assert matcher.first_divergence.request_prompt == "unrecorded"
    assert matcher.consumed_count() == 0


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


def test_client_has_the_live_client_signature(tmp_path):
    """complete() takes what DshModelClient.complete takes and nothing more: an
    optimizer call carrying a call_key fails here as it does live (#493 G30)."""
    import inspect

    from refactory_dsh.model import DshModelClient

    def params(fn):  # noqa: ANN001, ANN202
        return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]

    assert params(ReplayClient.complete) == params(DshModelClient.complete)
    client = _make_client(tmp_path)
    with pytest.raises(TypeError, match="call_key"):
        client.complete("test_prompt", system="test_sys", call_key=("analyst_failure", 1, 0, 0))


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

    with optimizer_call(("rank", 1, 0, 0)):
        client.complete("p1", system="sys1")
    client.complete("p2", system="sys2")  # a solver call: none in flight

    assert len(client.match_log) == 2
    assert client.match_log[0]["call_key"] == ("rank", 1, 0, 0)
    assert client.match_log[0]["index"] == 0
    assert client.match_log[1]["call_key"] is None


def test_client_match_log_counts_duplicate_keys(tmp_path):
    """The logged index is the recording served, so duplicates count up.

    The client reads it through ``CallMatcher.consumed_for`` rather than the
    private counter dict.
    """
    path = _make_calls_jsonl(tmp_path, [("sys", "p", "r1"), ("sys", "p", "r2")])
    client = ReplayClient(CallMatcher(path))

    assert client.complete("p", system="sys") == "r1"
    assert client.complete("p", system="sys") == "r2"
    assert [entry["index"] for entry in client.match_log] == [0, 1]


def test_matcher_consumed_for_is_public(tmp_path):
    path = _make_calls_jsonl(tmp_path, [("sys", "p", "r1"), ("sys", "p", "r2")])
    matcher = CallMatcher(path)

    assert matcher.consumed_for("sys", "p") == 0
    matcher.lookup("sys", "p")
    assert matcher.consumed_for("sys", "p") == 1
    assert matcher.consumed_for("sys", "absent") == 0


def test_matcher_recorded_prompts_for_a_system(tmp_path):
    path = _make_calls_jsonl(
        tmp_path, [("sysA", "p1", "r"), ("sysA", "p2", "r"), ("sysB", "p3", "r")],
    )
    matcher = CallMatcher(path)

    assert sorted(matcher.recorded_prompts("sysA")) == ["p1", "p2"]
    assert matcher.recorded_prompts("sysB") == ["p3"]
    assert matcher.recorded_prompts("sysC") == []


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
    # envs/base.py:265 hands the analysts the adapter's constant edit_budget.
    assert mapped["analyst_edit_budget"] == 4
    assert len(mapped) == 10  # exactly the options SkillOptLoop.configure takes


def test_config_mapping_carries_eval_test_when_a_recorded_it(tmp_path):
    """A's ``eval_test`` (trainer.py:2241) drives B's post-training evaluation."""
    base = {"num_epochs": 10, "early_stop_patience": 3, "early_stop_at_full_val": True,
            "edit_budget": 4, "min_edit_budget": 2, "seed": 42, "slow_update_samples": 20,
            "minibatch_size": 8, "merge_batch_size": 8}
    for value in (True, False):
        a_out = tmp_path / str(value)
        a_out.mkdir()
        (a_out / "config.json").write_text(json.dumps({**base, "eval_test": value}))
        assert load_a_config(a_out)["eval_test"] is value


def test_config_mapping_covers_every_configure_option():
    """Every option SkillOptLoop.configure reads is driven from A's config."""
    import inspect

    from refactory_packages.skillopt.loop import SkillOptLoop
    from refactory_packages.skillopt.replay_runner import (
        A_CONFIG_OPTIONAL,
        A_CONFIG_TO_STRATEGY_OPTION,
    )

    source = inspect.getsource(SkillOptLoop.configure)
    read = {
        line.split('options.get("')[1].split('"')[0]
        for line in source.splitlines()
        if 'options.get("' in line
    }
    # ``rollout_order`` is not a setting of A's: the runner passes the order A's
    # rollout completed in, read from A's step results (RecordedRolloutOrder).
    # ``claude_backend_wrapper`` is the runner's --claude-backend-wrapper flag.
    # ``knob`` is the runner's SKILL_KNOB, the LiveMath knob A's skill lives in;
    # ``knobs`` stays unset: A optimizes that one skill document.
    mapped = set(A_CONFIG_TO_STRATEGY_OPTION.values()) | set(A_CONFIG_OPTIONAL.values()) | {
        "analyst_edit_budget", "rollout_order", "patch_order", "claude_backend_wrapper", "knob",
        "knobs",
    }
    assert read == mapped, (
        f"configure reads {sorted(read)} but the replay drives {sorted(mapped)}"
    )


# ── import does not register ──────────────────────────────────────────────


def test_replay_import_does_not_register():
    """Importing replay.py does NOT add 'skillopt' to strategies.STRATEGIES.

    Checked in a fresh interpreter rather than with ``importlib.reload``: a
    reload rebinds the module's classes, so every ``except ReplayMismatch``
    elsewhere in the process would stop catching the exception this module
    raises, and the rest of this file would be testing two different classes.
    """
    import subprocess
    import sys

    code = (
        "from refactory_dsh import strategies\n"
        "before = set(strategies.STRATEGIES)\n"
        "import refactory_packages.skillopt.replay\n"
        "import refactory_packages.skillopt\n"
        "after = set(strategies.STRATEGIES)\n"
        "assert 'skillopt' not in after, 'importing registered skillopt'\n"
        "assert before == after, after - before\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr


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


# ── the duplicate port is gone ─────────────────────────────────────────────


def test_trainer_module_is_gone():
    """``skillopt/trainer.py`` — the standalone re-port of A — no longer exists.

    The replay must exercise arm B's ``SkillOptLoop``; a second port of the
    algorithm beside it can only be replayed instead of arm B.
    """
    import refactory_packages.skillopt as package

    package_dir = Path(package.__file__).resolve().parent
    assert not (package_dir / "trainer.py").exists()
    assert not (package_dir / "REPLAY_REPORT.md").exists()


def test_nothing_imports_the_trainer():
    """No module or test refers to the deleted trainer."""
    import refactory_packages.skillopt as package

    roots = [
        Path(package.__file__).resolve().parent,
        Path(__file__).resolve().parent,
    ]
    offenders = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text()
            for needle in ("skillopt.trainer", "SkillOptTrainer"):
                if needle in text and path.name != Path(__file__).name:
                    offenders.append(f"{path}: {needle}")
    assert not offenders, offenders


def test_trainer_import_fails():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("refactory_packages.skillopt.trainer")


# ── merge prompts are matched exactly ──────────────────────────────────────


MERGE_SYSTEM = "You are a skill-edit coordinator. Merge the patches."


def _patch(tag: str) -> dict:
    return {
        "reasoning": f"reasoning {tag}",
        "edits": [{"op": "append", "content": f"content {tag}"}],
    }


def _merge_prompt(patches: list[dict]) -> str:
    """The shape ``prompts.format_merge_user`` produces."""
    from refactory_packages.skillopt import prompts

    return prompts.format_merge_user("# Skill\nrules", patches, "", 1)


def _mismatch_error():
    """The live ``ReplayMismatch`` class, whatever module object holds it."""
    from refactory_packages.skillopt import replay as replay_mod

    return replay_mod.ReplayMismatch


def test_merge_prompt_formatting_is_matched_exactly(tmp_path):
    """A merge prompt is matched as B wrote it: a patch array encoded any other
    way (indent, ASCII escapes) is a divergence, never rewritten to match."""
    import json as _json

    from refactory_packages.skillopt import replay as replay_mod

    patches = [_patch("é one"), _patch("two")]
    recorded = _merge_prompt(patches)
    calls = _make_calls_jsonl(tmp_path, [(MERGE_SYSTEM, recorded, "MERGED")])
    client = replay_mod.ReplayClient(replay_mod.CallMatcher(calls))
    head = recorded[: recorded.index("[")]
    for encoded in (
        _json.dumps(patches, ensure_ascii=True, indent=2),
        _json.dumps(patches, ensure_ascii=False, indent=4),
    ):
        fresh = replay_mod.ReplayClient(replay_mod.CallMatcher(calls))
        with pytest.raises(_mismatch_error()), optimizer_call(("merge_failure", 1, 0, 1)):
            fresh.complete(head + encoded, system=MERGE_SYSTEM)
    with optimizer_call(("merge_failure", 1, 0, 1)):
        assert client.complete(recorded, system=MERGE_SYSTEM) == "MERGED"


def test_reordering_never_reaches_a_live_run():
    """The replay's order concessions never reach arm B's own modules, and no
    prompt rewriter exists any more: B's patch order is set at the source
    (``patch_order``), so its merge prompts are matched as written."""
    import inspect

    from refactory_packages.skillopt import analysts, loop, merge, replay, replay_runner

    assert not hasattr(replay, "MergePatchReorder")
    assert "reorder" not in inspect.getsource(replay_runner)
    assert "reorder" not in inspect.getsource(replay.ReplayClient)
    for module in (loop, analysts, merge):
        source = inspect.getsource(module)
        assert "RecordedRolloutOrder" not in source, module.__name__
        assert "RecordedPatchOrder" not in source, module.__name__
        assert "reorder" not in source, module.__name__
        assert "ReplayClient" not in source, module.__name__


def test_rollout_order_reaches_the_strategy_only_from_the_replay_runner():
    """A's completion order reaches arm B through ``SkillOptLoop``'s
    ``rollout_order`` option alone: off unless a callable is passed, and the
    replay runner is the only module in the packages that passes one."""
    import inspect
    from pathlib import Path as _Path

    import refactory_packages
    from refactory_packages.skillopt import replay_runner
    from refactory_packages.skillopt.loop import SkillOptLoop

    strategy = SkillOptLoop()
    strategy.configure()
    assert strategy.rollout_order is None
    with pytest.raises(TypeError):
        strategy.configure(rollout_order="results.jsonl")
    assert strategy.patch_order is None
    with pytest.raises(TypeError):
        strategy.configure(patch_order=["completion"])
    root = _Path(refactory_packages.__file__).parent
    for name in ("RecordedRolloutOrder(", "RecordedPatchOrder("):
        users = sorted(
            str(path.relative_to(root)) for path in root.rglob("*.py")
            if name in path.read_text() and path.name != "replay.py"
        )
        assert users == ["skillopt/replay_runner.py"], name
    source = inspect.getsource(replay_runner.run_replay)
    assert '"rollout_order": rollout_order' in source
    assert '"patch_order": patch_order' in source


def _write_results(a_out: Path, epoch: int, ids: list[str]) -> None:
    rollout = a_out / "steps" / f"step_{epoch:04d}" / "rollout"
    rollout.mkdir(parents=True, exist_ok=True)
    (rollout / "results.jsonl").write_text(
        "".join(json.dumps({"id": i, "hard": 0}) + "\n" for i in ids)
    )


def test_recorded_rollout_order_puts_records_in_completion_order(tmp_path):
    """rollout.py:418 writes results.jsonl in completion order, one step per epoch."""
    from refactory_packages.skillopt.replay import RecordedRolloutOrder

    _write_results(tmp_path, 2, ["c", "a", "b"])
    order = RecordedRolloutOrder(tmp_path)
    records = [{"id": "a", "n": 1}, {"id": "b", "n": 2}, {"id": "c", "n": 3}]
    assert order(2, records) == [records[2], records[0], records[1]]
    # A different item set, or no recorded step: unchanged, so content still diverges.
    assert order(2, records[:2]) == records[:2]
    assert order(3, records) == records
    assert [e["applied"] for e in order.applied] == [True, False, False]


# ── b_history_from_trace ───────────────────────────────────────────────────


def _trace_line(step_type: str, epoch: int, **fields) -> dict:
    return {
        "type": "decision", "timestamp": 1.0,
        "epoch": epoch, "step_type": step_type, **fields,
    }


def _known_trace() -> list[dict]:
    """Two epochs of arm B's trace: epoch 1 accepts, epoch 2 skips and stops."""
    merged = {"reasoning": "m", "edits": [{"op": "append", "content": "X"}]}
    ranked = [{"op": "append", "content": "X"}]
    return [
        _trace_line("initial_val", 0, score=0.25, skill_hash="a" * 64),
        _trace_line(
            "train_rollout", 1, n_records=4, epoch_seed=1043,
            rollout_ids=["i1", "i2", "i3", "i4"], rollout_hard=0.5,
        ),
        {"type": "model_call", "timestamp": 1.0, "epoch": 1,
         "step_type": "analyst_failure", "system_prompt": "s",
         "user_prompt": "u", "raw_output": "o"},
        _trace_line("analyst_failure", 1, group_index=0, record_ids=["i1"], n_edits=1),
        _trace_line("analyst_failure", 1, group_index=1, record_ids=["i2"], n_edits=0),
        _trace_line("analyst_success", 1, group_index=0, record_ids=["i3"], n_edits=2),
        _trace_line("merged_patch", 1, patch=merged, n_edits=1),
        _trace_line("ranked_edits", 1, budget_step=1, edit_budget=2, n_edits=1, edits=ranked),
        _trace_line(
            "patch_applied", 1, n_edits=1,
            edit_report={"outcomes": [
                {"index": 0, "kind": "append", "target": "", "content_preview": "X",
                 "status": "applied_append"},
                {"index": 1, "kind": "replace", "target": "T", "content_preview": "Y",
                 "status": "skipped_replace_target_not_found"},
            ], "noop": False},
        ),
        _trace_line("val_evaluate", 1, skill_hash="b" * 64, score=0.75),
        _trace_line("val_gate_accept", 1, cand_score=0.75, current_score=0.25),
        _trace_line("slow_update_placeholder", 1),
        _trace_line("meta_skill", 1, action="skip_first_epoch"),
        _trace_line(
            "train_rollout", 2, n_records=4, epoch_seed=2043,
            rollout_ids=["i2", "i1", "i4", "i3"], rollout_hard=0.75,
        ),
        _trace_line("analyst_failure", 2, group_index=0, record_ids=["i1"], n_edits=0),
        _trace_line("no_edits_skip", 2),
        _trace_line("slow_update_sample", 2, sample_ids=["i1", "i2"]),
        _trace_line("slow_update", 2, action="force_accept", guidance_length=12, n_pairs=2),
        _trace_line("meta_skill", 2, action="write_meta_skill", meta_skill_length=4),
        _trace_line("early_stop", 2, reason="patience"),
    ]


def test_b_history_from_trace_maps_a_known_trace():
    """A known trace maps to exactly the entries ``compare_runs`` indexes."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())

    assert [(e["epoch"], e["step"]) for e in history] == [(1, 1), (2, 2)]

    first = history[0]
    assert first["rollout_hard"] == 0.5
    assert first["rollout_ids"] == ["i1", "i2", "i3", "i4"]
    assert first["rollout_n"] == 4
    # Only the groups that proposed edits count as patches.
    assert first["n_failure_patches"] == 1
    assert first["n_success_patches"] == 1
    assert first["n_patches"] == 2
    assert first["patch_files"] == [
        "minibatch_fail_000.json", "minibatch_succ_000.json",
    ]
    assert first["merged_patch"] == {
        "reasoning": "m", "edits": [{"op": "append", "content": "X"}],
    }
    assert first["n_edits_merged"] == 1
    assert first["ranked_edits"] == [{"op": "append", "content": "X"}]
    assert first["n_edits_ranked"] == 1
    assert [row["status"] for row in first["edit_apply_report"]] == [
        "applied_append", "skipped_replace_target_not_found",
    ]
    assert first["edit_apply_summary"] == {
        "total": 2, "applied": 1, "skipped": 1, "errors": 0,
    }
    assert first["candidate_hash"] == "b" * 64
    assert first["selection_hard"] == 0.75
    assert first["action"] == "accept_new_best"
    assert first["slow_update"] == {"action": "inject_placeholder"}
    assert first["meta_skill"] == {"action": "skip_first_epoch"}
    assert first["early_stop"] is False

    second = history[1]
    assert second["action"] == "skip_no_patches"
    # Nothing was merged, ranked or applied, so those fields stay absent.
    assert second["n_patches"] == 0
    assert second["merged_patch"] is None
    assert second["n_edits_merged"] is None
    assert second["ranked_edits"] is None
    assert second["edit_apply_report"] is None
    assert second["edit_apply_summary"] is None
    assert second["candidate_hash"] is None
    assert second["selection_hard"] is None
    assert second["slow_update"] == {"action": "force_accept"}
    assert second["meta_skill"] == {"action": "write_meta_skill"}
    assert second["early_stop"] is True


def test_b_history_from_trace_never_invents_a_field():
    """An epoch arm B barely recorded leaves every unknown field ``None``."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace([_trace_line("train_rollout", 1, n_records=0)])

    assert len(history) == 1
    entry = history[0]
    for field in (
        "rollout_hard", "rollout_ids", "n_patches", "patch_files", "merged_patch",
        "n_edits_merged", "ranked_edits", "n_edits_ranked", "edit_apply_report",
        "edit_apply_summary", "candidate_hash", "selection_hard", "action",
        "slow_update", "meta_skill",
    ):
        assert entry[field] is None, field


def test_b_history_from_trace_second_accept_is_not_a_new_best():
    """An accept that does not beat the best so far is ``accept``, not best."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    trace = [
        _trace_line("initial_val", 0, score=0.9),
        _trace_line("val_evaluate", 1, skill_hash="c" * 64, score=0.5),
        _trace_line("val_gate_accept", 1, cand_score=0.5, current_score=0.4),
    ]
    assert b_history_from_trace(trace)[0]["action"] == "accept"


def test_b_history_feeds_compare_runs(tmp_path):
    """The mapped history is what ``compare_runs`` reads: a faithful A matches."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)

    report = compare_runs(a_out, history)

    assert report["first_divergence"] is None, report["first_divergence"]
    assert report["summary"]["diverged_epochs"] == 0, format_report(report)


# ── a synthetic A out/ built from a B history ──────────────────────────────


def _a_out_from_b_history(
    tmp_path: Path, b_history: list[dict], *, num_epochs: int,
    tweak: dict | None = None, reverse_rollout: bool = True,
) -> Path:
    """Write an A ``out/`` directory that agrees with *b_history*.

    Used to check the comparison's plumbing end to end: with ``tweak`` one
    field of A's first entry is changed, so the divergence the report must find
    is known in advance.
    """
    a_out = tmp_path / "a_out"
    for sub in ("skills", "meta_skill", "slow_update", "steps"):
        (a_out / sub).mkdir(parents=True, exist_ok=True)

    history = []
    for index, entry in enumerate(b_history):
        a_entry = {
            "step": entry["step"],
            "epoch": entry["epoch"],
            "step_in_epoch": 0,
            "rollout_hard": entry["rollout_hard"],
            "rollout_n": entry["rollout_n"],
            "n_patches": entry["n_patches"],
            "n_failure_patches": entry["n_failure_patches"],
            "n_success_patches": entry["n_success_patches"],
            "n_edits_merged": entry["n_edits_merged"],
            "n_edits_ranked": entry["n_edits_ranked"],
            "candidate_hash": (entry["candidate_hash"] or "")[:16] or None,
            "edit_apply_summary": entry["edit_apply_summary"],
            "selection_hard": entry["selection_hard"],
            "action": entry["action"],
        }
        if tweak and index == 0:
            a_entry.update(tweak)
        history.append(a_entry)
    (a_out / "history.json").write_text(json.dumps(history))
    (a_out / "config.json").write_text(json.dumps({"num_epochs": num_epochs}))

    for entry in b_history:
        step_dir = a_out / "steps" / f"step_{entry['step']:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if entry["rollout_ids"] is not None:
            rollout = step_dir / "rollout"
            rollout.mkdir(exist_ok=True)
            # A writes results.jsonl in rollout completion order; reversing it
            # here keeps the comparison honest about ignoring that order. A
            # recording replayed through ``run_replay`` keeps B's own order: the
            # replay hands it to the analysts as A's (RecordedRolloutOrder).
            ids = entry["rollout_ids"]
            rollout.joinpath("results.jsonl").write_text(
                "\n".join(
                    json.dumps({"id": item_id})
                    for item_id in (reversed(ids) if reverse_rollout else ids)
                ) + "\n"
            )
        if entry["patch_files"]:
            patches = step_dir / "patches"
            patches.mkdir(exist_ok=True)
            for name in entry["patch_files"]:
                patches.joinpath(name).write_text("{}")
        if entry["merged_patch"] is not None:
            step_dir.joinpath("merged_patch.json").write_text(
                json.dumps(entry["merged_patch"])
            )
        if entry["ranked_edits"] is not None:
            step_dir.joinpath("ranked_edits.json").write_text(
                json.dumps({"reasoning": "r", "ranking_details": [],
                            "edits": entry["ranked_edits"]})
            )
        if entry["edit_apply_report"] is not None:
            step_dir.joinpath("edit_apply_report.json").write_text(
                json.dumps(entry["edit_apply_report"])
            )
        for field, directory, filename in (
            ("slow_update", "slow_update", "slow_result.json"),
            ("meta_skill", "meta_skill", "meta_skill_result.json"),
        ):
            value = entry[field]
            if value is None:
                continue
            epoch_dir = a_out / directory / f"epoch_{entry['epoch']:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            # A's own result files carry bookkeeping B has no equivalent for.
            epoch_dir.joinpath(filename).write_text(
                json.dumps({**value, "epoch": entry["epoch"], "time_s": 1.5})
            )
    return a_out


def test_a_bookkeeping_only_difference_is_not_a_divergence(tmp_path):
    """A's ``time_s`` in a slow-update result is not a disagreement with B."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)

    report = compare_runs(a_out, history)
    per_epoch = {ep["epoch"]: ep for ep in report["per_epoch"]}
    assert per_epoch[2]["slow_update"]["match"] is True
    assert per_epoch[1]["meta_skill"]["match"] is True


def test_rollout_order_alone_is_not_a_divergence(tmp_path):
    """A's rollout order is a race (``workers: 24``), so only the ids compare."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)

    report = compare_runs(a_out, history)
    per_epoch = {ep["epoch"]: ep for ep in report["per_epoch"]}
    assert per_epoch[1]["reflection_groups"]["match"] is True


def test_a_different_rollout_set_is_a_divergence(tmp_path):
    """A different set of rolled-out ids still fails, order-insensitive or not."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)
    results = a_out / "steps" / "step_0001" / "rollout" / "results.jsonl"
    results.write_text(
        "\n".join(json.dumps({"id": i}) for i in ["i1", "i2", "i3", "i9"]) + "\n"
    )

    report = compare_runs(a_out, history)
    assert report["first_divergence"]["field"] == "reflection_groups"


def test_a_different_patch_group_is_a_divergence(tmp_path):
    """Which analyst group produced a patch is compared, not just how many."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)
    patches = a_out / "steps" / "step_0001" / "patches"
    patches.joinpath("minibatch_fail_000.json").unlink()
    patches.joinpath("minibatch_fail_001.json").write_text("{}")

    report = compare_runs(a_out, history)
    assert report["first_divergence"]["field"] == "reflection_groups"


def test_ranked_edits_compare_structurally(tmp_path):
    """A changed ranked edit is found even when the count still agrees."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)
    a_out.joinpath("steps", "step_0001", "ranked_edits.json").write_text(
        json.dumps({"reasoning": "r", "ranking_details": [],
                    "edits": [{"op": "append", "content": "DIFFERENT"}]})
    )

    report = compare_runs(a_out, history)
    assert report["first_divergence"]["field"] == "ranked_edits"


def test_ranked_edits_match_across_a_s_payload_shape(tmp_path):
    """A's ranked payload is a dict and B's a list; equal edits still match.

    Without the shape normalisation every epoch would report a ranked-edits
    divergence, whatever arm B selected.
    """
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)

    report = compare_runs(a_out, history)
    per_epoch = {ep["epoch"]: ep for ep in report["per_epoch"]}
    assert per_epoch[1]["ranked_edits"]["match"] is True


def test_merged_edits_compare_structurally(tmp_path):
    """A changed merged edit is found even when the count still agrees."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)
    a_out.joinpath("steps", "step_0001", "merged_patch.json").write_text(
        json.dumps({"reasoning": "m",
                    "edits": [{"op": "append", "content": "DIFFERENT"}]})
    )

    report = compare_runs(a_out, history)
    assert report["first_divergence"]["field"] == "merged_edits"


# ── run_replay drives arm B ────────────────────────────────────────────────

LIVEMATH_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "livemath"
LIVEMATH_SPLIT = LIVEMATH_FIXTURES / "split"

#: The strategy options the recording pass and the replay pass both run with,
#: in A's own ``config.json`` spelling so ``load_a_config`` maps them back.
RECORDING_A_CONFIG = {
    "num_epochs": 2,
    "early_stop_patience": 0,
    "early_stop_at_full_val": False,
    "edit_budget": 2,
    "min_edit_budget": 1,
    "seed": 42,
    "slow_update_samples": 2,
    "minibatch_size": 8,
    "merge_batch_size": 8,
    "use_slow_update": True,
    "use_meta_skill": True,
    "skill_update_mode": "patch",
    "failure_only": False,
    "use_gate": True,
    "steps_per_epoch": 1,
    "batches_per_epoch": 1,
    "accumulation": 1,
    "train_size": 2,
}


class _RecordingClient:
    """Answers every call with a canned reply and records what it was asked.

    Mirrors ``ReplayClient``'s protocol so the recording pass and the replay
    pass present the same seam to arm B.
    """

    def __init__(self) -> None:
        self.recorded: list[tuple[str, str, str]] = []
        #: (step type, prompt) exactly as arm B sent each call.
        self.sent: list[tuple[str, str]] = []

    def complete(self, prompt, *, system="", workdir=None, cwd=None, env=None, confine=False):  # noqa: ANN001
        call_key = current_call_key()
        step_type = str(call_key[0]) if call_key else ""
        result = self._reply(step_type)
        self.sent.append((step_type, prompt))
        # A's backend frames the bare user message (_make_calls_jsonl does it
        # again), so record the solve node's prompt -- and an optimizer prompt
        # arm B framed (claude_backend_wrapper) -- without its framing.
        prompt = normalize_recorded_prompt(prompt) if prompt.startswith("History:\n") else prompt
        self.recorded.append((system, prompt, result))
        return result

    @staticmethod
    def _reply(step_type: str) -> str:
        if step_type.startswith("analyst_"):
            return json.dumps({
                "batch_size": 2,
                "patch": {
                    "reasoning": f"{step_type} reasoning",
                    "edits": [{"op": "append", "content": "## Rule\nCheck twice."}],
                },
            })
        if step_type.startswith("merge_"):
            return json.dumps({
                "reasoning": "merged",
                "edits": [{"op": "append", "content": "## Merged\nCombined."}],
            })
        if step_type == "rank":
            return json.dumps({"reasoning": "ranked", "selected_indices": [0]})
        if step_type == "slow_update":
            return json.dumps({"reasoning": "r", "slow_update_content": "Guidance."})
        if step_type == "meta_skill":
            return json.dumps({"reasoning": "r", "meta_skill_content": "CONCISE"})
        # The solver agent node carries no call key.
        return "<answer>B</answer>"

    def for_node(self, node_id, role=None):  # noqa: ANN001
        return self

    def with_tool_policy(self, policy):  # noqa: ANN001
        return self

    def with_wall(self, seconds):  # noqa: ANN001
        return self

    def route(self):
        return {"worker_backend": "dsh"}


#: The seed skill the fixture records with, and stores in the recording. Every solver
#: and analyst prompt embeds it, so recording and replay must use the same text --
#: the first real s42 replay matched 0 of 1280 calls because the runner loaded A's
#: seed and never passed it to the loop.
FIXTURE_SEED_SKILL = "# Seed heuristics\n\n- Compare every option before committing.\n"


def _b_config(runs_root: Path, parts_path: Path, max_epochs: int):
    from refactory_dsh.loop import LoopConfig
    from refactory_packages.skillopt.replay_runner import ALLOWED_OPS, SKILL_KNOB

    return LoopConfig(
        knobs={SKILL_KNOB: FIXTURE_SEED_SKILL},
        epochs=max_epochs,
        batch_size=1,
        optimizer="random",
        seed=RECORDING_A_CONFIG["seed"],
        runs_root=runs_root,
        allowed_ops=ALLOWED_OPS,
        strategy="skillopt",
        parts_manifest=str(parts_path),
        max_evaluations=4 * max_epochs + 16,
        # As the replay runs it: no holdout audit (run_replay's LoopConfig).
        holdout=0.0,
    )


def _record_arm_b(
    tmp_path: Path, *, claude_backend_wrapper: bool = False, client: Any = None,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Run arm B once with a canned client and return (recordings, b history).

    The recording pass goes through the same public path the replay does, so
    the recordings are exactly the calls the replay has to answer.
    """
    from refactory_dsh import strategies
    from refactory_packages.livemath.manifest import build_parts_manifest
    from refactory_packages.skillopt.replay_runner import (
        b_history_from_trace, load_a_config, read_trace,
    )

    a_out = tmp_path / "record_config"
    a_out.mkdir()
    (a_out / "config.json").write_text(json.dumps(RECORDING_A_CONFIG))
    options = load_a_config(a_out)
    if claude_backend_wrapper:
        options["claude_backend_wrapper"] = True

    parts_path = build_parts_manifest(LIVEMATH_SPLIT, tmp_path / "record")
    runs_root = tmp_path / "record" / "run"
    client = client if client is not None else _RecordingClient()
    previous = os.environ.get("LIVEMATH_DATA_PATH")
    os.environ["LIVEMATH_DATA_PATH"] = str(LIVEMATH_FIXTURES)
    try:
        loop = strategies.create(
            "skillopt", "livemath/solver",
            _b_config(runs_root, parts_path, options["max_epochs"]),
            strategy_options=options, project=Path("."), model=client,
        )
        loop.run()
    finally:
        if previous is None:
            os.environ.pop("LIVEMATH_DATA_PATH", None)
        else:
            os.environ["LIVEMATH_DATA_PATH"] = previous
    return client.recorded, b_history_from_trace(read_trace(runs_root))


def _recording_dir(
    tmp_path: Path,
    recordings: list[tuple[str, str, str]],
    b_history: list[dict],
    *,
    tweak: dict | None = None,
) -> Path:
    """Assemble a recording directory arm B can be replayed against."""
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir()
    _make_calls_jsonl(recording_dir, recordings)
    a_out = _a_out_from_b_history(
        tmp_path, b_history, num_epochs=RECORDING_A_CONFIG["num_epochs"], tweak=tweak,
        reverse_rollout=False,
    )
    a_out.rename(recording_dir / "out")
    (recording_dir / "out" / "config.json").write_text(json.dumps(RECORDING_A_CONFIG))
    # SkillOpt writes the seed it started from to out/skills/skill_v0000.md, and the
    # replay seeds arm B's skill knob from it.
    skills = recording_dir / "out" / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    (skills / "skill_v0000.md").write_text(FIXTURE_SEED_SKILL)
    return recording_dir


@pytest.fixture(scope="module")
def _arm_b_recording(tmp_path_factory):
    """One recording pass, shared by the run_replay tests (it runs the graph)."""
    tmp_path = tmp_path_factory.mktemp("record")
    recordings, b_history = _record_arm_b(tmp_path)
    assert b_history, "the recording pass produced no B history"
    return recordings, b_history


def test_run_replay_drives_arm_b(tmp_path, monkeypatch, _arm_b_recording):
    """``run_replay`` reaches arm B's ``SkillOptLoop`` through the public path."""
    from refactory_packages.skillopt.loop import SkillOptLoop
    from refactory_packages.skillopt.replay_runner import MODE_ID, run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)

    report = run_replay(
        recording_dir, tmp_path / "out",
        split_dir=LIVEMATH_SPLIT, data_path=LIVEMATH_FIXTURES,
    )

    # The strategy came from package discovery and is arm B's class.
    assert report["strategy"]["name"] == "skillopt"
    assert report["strategy"]["class"] == (
        f"{SkillOptLoop.__module__}.{SkillOptLoop.__qualname__}"
    )
    assert report["mode_id"] == MODE_ID == "livemath/solver"
    # Only SkillOptLoop.strategy() writes skillopt_trace.jsonl.
    assert report["strategy_ran"] is True
    assert (Path(report["runs_root"]) / "skillopt_trace.jsonl").exists()
    assert report["replay_status"] == "completed", report["mismatch"]
    assert report["parts"] == {
        "manifest": report["parts"]["manifest"], "train": 2, "val": 1, "test": 2,
    }

    # Arm B asked for every recorded call and nothing else.
    assert report["calls_matched"] == report["calls_total"] > 0
    assert report["unused_recordings"] == []

    # The history the comparison reads is arm B's real work, not placeholders.
    assert report["b_history"], report
    first = report["b_history"][0]
    train_ids = json.loads((LIVEMATH_SPLIT / "train" / "items.json").read_text())
    assert sorted(first["rollout_ids"]) == sorted(str(i["id"]) for i in train_ids)
    assert first["rollout_hard"] is not None
    assert first["merged_patch"] is not None
    assert first["ranked_edits"]
    assert first["edit_apply_report"]
    assert first["candidate_hash"]
    # Epoch 1 has no previous epoch, and arm B records the skip the way A does.
    assert first["meta_skill"] == {"action": "skip_first_epoch"}
    assert first["slow_update"] == {"action": "inject_placeholder"}

    # Replaying arm B against its own recordings diverges nowhere.
    assert report["comparison"]["first_divergence"] is None, format_report(
        report["comparison"]
    )
    assert report["comparison"]["summary"]["diverged_epochs"] == 0
    # And the strategy is not in STRATEGIES: discovery, never a hand register().
    from refactory_dsh import strategies
    assert "skillopt" not in strategies.STRATEGIES


def test_run_replay_report_is_written(tmp_path, monkeypatch, _arm_b_recording):
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)

    report = run_replay(
        recording_dir, tmp_path / "out",
        split_dir=LIVEMATH_SPLIT, data_path=LIVEMATH_FIXTURES,
    )

    written = json.loads((tmp_path / "out" / "replay_report.json").read_text())
    assert written["replay_status"] == report["replay_status"]
    assert written["b_history"] == report["b_history"]


def test_run_replay_reports_a_planted_divergence(tmp_path, monkeypatch, _arm_b_recording):
    """A changed A value is reported as a mismatch carrying both values."""
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    b_val = b_history[0]["selection_hard"]
    planted = (b_val or 0.0) + 0.5
    recording_dir = _recording_dir(
        tmp_path, recordings, b_history, tweak={"selection_hard": planted},
    )

    report = run_replay(
        recording_dir, tmp_path / "out",
        split_dir=LIVEMATH_SPLIT, data_path=LIVEMATH_FIXTURES,
    )

    divergence = report["comparison"]["first_divergence"]
    assert divergence is not None, format_report(report["comparison"])
    assert divergence["field"] == "val_scores"
    assert divergence["epoch"] == b_history[0]["epoch"]
    assert divergence["a_value"] == planted
    assert divergence["b_value"] == b_val


def test_run_replay_refuses_a_recording_arm_b_cannot_match(tmp_path, monkeypatch, _arm_b_recording):
    """A config arm B has no option for is reported, not quietly replayed."""
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)
    config = json.loads((recording_dir / "out" / "config.json").read_text())
    config["use_slow_update"] = False
    (recording_dir / "out" / "config.json").write_text(json.dumps(config))

    report = run_replay(
        recording_dir, tmp_path / "out",
        split_dir=LIVEMATH_SPLIT, data_path=LIVEMATH_FIXTURES,
    )

    assert report["replay_status"] == "error"
    assert any("use_slow_update" in line for line in report["unsupported_a_settings"])
    assert report["strategy_ran"] is False


def test_run_replay_refuses_a_recording_without_its_seed_skill(tmp_path, monkeypatch, _arm_b_recording):
    """No seed skill, no replay: every prompt embeds it, so nothing could match.

    The first real s42 replay loaded A's seed and never passed it to the loop, and
    compared 0 of 1280 calls without saying why. A seed that cannot be loaded is now
    refused before arm B runs.
    """
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)
    (recording_dir / "out" / "skills" / "skill_v0000.md").unlink()
    config = json.loads((recording_dir / "out" / "config.json").read_text())
    config["skill_init"] = "nowhere/initial.md"
    (recording_dir / "out" / "config.json").write_text(json.dumps(config))

    report = run_replay(recording_dir, tmp_path / "out",
                        split_dir=LIVEMATH_SPLIT, data_path=LIVEMATH_FIXTURES)

    assert report["replay_status"] == "error", report["replay_status"]
    assert "seed skill could not be loaded" in report["mismatch"]["error"]
    assert report["strategy_ran"] is False


def test_run_replay_with_a_different_split_diverges(tmp_path, monkeypatch, _arm_b_recording):
    """A split that is not the one A ran is refused, naming A's train size."""
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)

    other_split = tmp_path / "other_split"
    for part in ("train", "val", "test"):
        (other_split / part).mkdir(parents=True)
    (other_split / "train" / "items.json").write_text(json.dumps([{"id": "202501:1"}]))
    (other_split / "val" / "items.json").write_text(json.dumps([{"id": "202503:1"}]))
    (other_split / "test" / "items.json").write_text(json.dumps([{"id": "202503:2"}]))

    report = run_replay(
        recording_dir, tmp_path / "out",
        split_dir=other_split, data_path=LIVEMATH_FIXTURES,
    )

    assert report["replay_status"] == "error"
    assert "train_size=2" in report["mismatch"]["error"]
    assert report["strategy_ran"] is False


def test_run_replay_reports_a_missing_recording_as_a_mismatch(tmp_path, monkeypatch, _arm_b_recording):
    """A call A never made is reported as a mismatch naming the diverging prompt.

    This is the other half of divergence detection: the comparison finds a
    different decision, and the matcher finds a different *call*.
    """
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    # Drop the analyst call arm B makes first: a strategy-level call, so the
    # mismatch surfaces from the strategy rather than inside a graph node.
    dropped = next(
        index for index, (_, prompt, _) in enumerate(recordings)
        if "## Current Skill" in prompt
    )
    thinned = recordings[:dropped] + recordings[dropped + 1:]
    recording_dir = _recording_dir(tmp_path, thinned, b_history)

    report = run_replay(
        recording_dir, tmp_path / "out",
        split_dir=LIVEMATH_SPLIT, data_path=LIVEMATH_FIXTURES,
    )

    assert report["replay_status"] in ("mismatch", "error"), report["replay_status"]
    assert report["mismatch"] is not None
    assert report["calls_total"] == len(thinned)

# ── A's environment events: failed attempts and its own timeouts ──────────


def _calls_with_rc(tmp_path: Path, entries: list[tuple[str, str, str, int]]) -> Path:
    """calls.jsonl from (system, prompt, result, rc); rc != 0 is a failed attempt."""
    path = tmp_path / "calls.jsonl"
    lines = []
    for i, (system, prompt, result, rc) in enumerate(entries):
        lines.append(json.dumps({
            "at": 1000.0 + i, "rc": rc, "system": system, "prompt": _wrap_prompt(prompt),
            "stdout": json.dumps({"result": result, "is_error": rc != 0}),
        }))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_failed_attempts_replay_as_a_retries(tmp_path):
    """claude_backend.py:309-324: failed attempts before a success are retries;
    attempts used up (or no success left) fail the call with A's message."""
    from refactory_packages.skillopt.replay import ReplayedCallFailure

    matcher = CallMatcher(_calls_with_rc(tmp_path, [
        ("s", "p", "API Error", 1), ("s", "p", "API Error", 1), ("s", "p", "ok", 0),
        ("s", "q", "API Error", 1), ("s", "q", "API Error", 1), ("s", "q", "late ok", 0),
        ("s", "r", "API Error", 1),
    ]))
    assert matcher.lookup("s", "p", attempts=5) == "ok"
    with pytest.raises(ReplayedCallFailure, match="failed after 2 retries: Claude CLI exited with code 1"):
        matcher.lookup("s", "q", attempts=2)
    assert matcher.lookup("s", "q", attempts=2) == "late ok"
    with pytest.raises(ReplayedCallFailure, match="failed after 3 retries"):
        matcher.lookup("s", "r", attempts=3)
    assert matcher.first_divergence is None
    assert matcher.failed_attempts_replayed == 5
    assert matcher.consumed_count() == 7


def test_late_answer_times_the_call_out_and_is_withheld(tmp_path):
    """rollout.py:394-432: A reported the item as a task timeout; its answer came
    later. B's call times out (as refactory's CompletionTimeout) and the late
    answer is never served, nor counted as matched."""
    from refactory_dsh.model import CompletionTimeout

    matcher = CallMatcher(_calls_with_rc(tmp_path, [
        ("s", "p", "API Error", 1), ("s", "p", "late answer", 0), ("s", "p", "second", 0),
    ]))
    matcher.mark_late([1])
    with pytest.raises(CompletionTimeout, match="calls.jsonl line 1") as exc_info:
        matcher.lookup("s", "p", attempts=5)
    # B's walker records it as it records a live timeout ("<node>: <type>: <msg>"),
    # and LiveMath renders that as SkillOpt's task timeout.
    from refactory_packages.livemath.skillopt_view import error_fail_reason

    event = exc_info.value
    assert error_fail_reason({
        "error_kind": "RuntimeError", "error_text": f"solve: {type(event).__name__}: {event}",
    }) == "task-timeout-1860s"
    assert matcher.lookup("s", "p", attempts=5) == "second"
    assert matcher.first_divergence is None
    assert [entry["line"] for entry in matcher.late_answers()] == [1]
    assert matcher.consumed_count() == 2  # the failed attempt and "second"


def test_replay_client_gives_solver_calls_five_attempts_and_optimizer_calls_three(tmp_path, _no_retry_waits):
    """rollout.py:227 asks with retries=5; every optimizer call with retries=3,
    made by ``trace.ask`` one attempt per ``complete``: each of B's attempts
    replays one of A's, so A's three failures fail B's call and A's next
    answer goes to B's next call."""
    from refactory_packages.skillopt.replay import ReplayedCallFailure
    from refactory_packages.skillopt.trace import TraceRecorder, ask

    entries = [("s", "solver", "E", 1)] * 4 + [("s", "solver", "ok", 0)]
    entries += [("s", "opt", "E", 1)] * 3 + [("s", "opt", "ok", 0)]
    entries += [("s", "opt2", "E", 1), ("s", "opt2", "ok2", 0)]
    entries += [("s", "raw", "E", 1), ("s", "raw", "raw ok", 0)]
    client = ReplayClient(CallMatcher(_calls_with_rc(tmp_path, entries)))
    assert client.complete("solver", system="s") == "ok"
    assert client.matcher.failed_attempts_replayed == 4
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    assert ask(client, trace, "rank", "opt", system="s", call_key=("rank", 1, 0, 0)) is None
    assert client.matcher.failed_attempts_replayed == 4 + 3
    assert ask(client, trace, "rank", "opt", system="s", call_key=("rank", 1, 0, 1)) == "ok"
    assert ask(client, trace, "rank", "opt2", system="s", call_key=("rank", 1, 0, 2)) == "ok2"
    assert client.match_log[-2]["a_event"].startswith("ReplayedCallFailure")
    calls = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [(c["call_key"][3], c["attempt"], "error" in c) for c in calls] == [
        (0, 1, True), (0, 2, True), (0, 3, True), (1, 1, False), (2, 1, True), (2, 2, False),
    ]
    assert _no_retry_waits == [1, 2, 4, 1]
    # Outside trace.ask, one optimizer complete replays exactly one of A's attempts.
    with pytest.raises(ReplayedCallFailure), optimizer_call(("rank", 1, 0, 3)):
        client.complete("raw", system="s")
    with optimizer_call(("rank", 1, 0, 3)):
        assert client.complete("raw", system="s") == "raw ok"
    assert client.matcher.first_divergence is None
    assert client.matcher.consumed_count() == len(entries)


def test_late_answer_lines_found_from_a_timeout_result(tmp_path):
    """A's timed-out item: its prediction files name the call, the conversation
    names the late recording; an event without them is unresolved, not guessed."""
    import hashlib

    from refactory_packages.skillopt.replay import late_answer_lines

    calls = _calls_with_rc(tmp_path, [
        ("SYS", "USER", "early", 0), ("SYS", "USER", "<answer>D</answer>", 0),
    ])
    rollout = tmp_path / "out" / "slow_update" / "epoch_02" / "rollout_prev"
    item = rollout / "predictions" / ("_mapped-" + hashlib.sha256(b"202512:33").hexdigest())
    item.mkdir(parents=True)
    (item / "target_system_prompt.txt").write_text("SYS\n")
    (item / "target_user_prompt.txt").write_text("USER")
    (item / "conversation.json").write_text(json.dumps([
        {"type": "message", "turn": 1, "content": "<answer>D</answer>"},
        {"role": "system", "content": "[EVALUATION RESULT]"},
    ]))
    (rollout / "results.jsonl").write_text(
        json.dumps({"id": "202512:33", "fail_reason": "task-timeout-1860s"}) + "\n"
        + json.dumps({"id": "202512:34", "fail_reason": "task-timeout-1860s"}) + "\n"
        + json.dumps({"id": "202512:35", "fail_reason": "MCQ=0: x"}) + "\n"
    )
    found = late_answer_lines(tmp_path / "out", CallMatcher(calls))
    assert found["lines"] == [1]
    assert found["events"][0]["rollout"] == "slow_update/epoch_02/rollout_prev"
    assert [e["id"] for e in found["unresolved"]] == ["202512:34"]


def test_events_are_conceded_by_the_replay_runner_alone():
    """Only the replay runner marks A's late answers; no arm-B module knows them."""
    import inspect
    from pathlib import Path as _Path

    import refactory_packages
    from refactory_packages.livemath import mode, ops, skillopt_view
    from refactory_packages.skillopt import analysts, env, loop, merge, prompts, trace

    for module in (loop, analysts, merge, prompts, trace, env, mode, ops, skillopt_view):
        source = inspect.getsource(module)
        for name in ("mark_late", "late_answer", "A_TIMEOUT_MESSAGE", "ReplayedCallFailure"):
            assert name not in source, (module.__name__, name)
    root = _Path(refactory_packages.__file__).parent
    users = sorted(
        str(path.relative_to(root)) for path in root.rglob("*.py")
        if ".mark_late(" in path.read_text() and path.name != "replay.py"
    )
    assert users == ["skillopt/replay_runner.py"]


def test_run_replay_refuses_a_different_exec_timeout(tmp_path):
    """A's exec_timeout cross-checks B's LiveMath solve timeout (livemath/mode.py)."""
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.skillopt.replay_runner import unsupported_a_settings

    (tmp_path / "config.json").write_text(json.dumps({"exec_timeout": Solver.solve_timeout_s}))
    assert unsupported_a_settings(tmp_path) == []
    (tmp_path / "config.json").write_text(json.dumps({"exec_timeout": 300}))
    assert any("exec_timeout=300" in line for line in unsupported_a_settings(tmp_path))


def test_a_fault_in_the_replay_is_kept_for_the_report(tmp_path):
    """B swallows a failed optimizer call like SkillOpt; the replay must not."""
    client = ReplayClient(CallMatcher(_calls_with_rc(tmp_path, [("s", "p", "ok", 0)])))

    def broken(system, prompt, **kwargs):  # noqa: ANN001, ANN003
        raise TypeError("bug in the replay")

    client.matcher.lookup = broken  # type: ignore[method-assign]
    with pytest.raises(TypeError), optimizer_call(("rank", 1, 0, 0)):
        client.complete("p", system="s")
    assert client.harness_error == "TypeError: bug in the replay"


def test_recorded_patch_order_reads_a_failed_merge_prompt(tmp_path):
    """A's order comes from its own merge prompt, even one whose call failed
    (calls.jsonl rc=1): reflect.py:643-650 appended patches as they completed."""
    from refactory_packages.skillopt import prompts as skillopt_prompts
    from refactory_packages.skillopt.replay import RecordedPatchOrder

    p1 = {"reasoning": "one", "edits": [{"op": "append", "content": "a"}]}
    p2 = {"reasoning": "two", "edits": [{"op": "append", "content": "b"}]}
    a_prompt = skillopt_prompts.format_merge_user("SKILL", [p2, p1], "", 1)
    matcher = CallMatcher(_calls_with_rc(tmp_path, [
        (skillopt_prompts.MERGE_FAILURE_SYSTEM, a_prompt, "API Error", 1),
    ]))
    order = RecordedPatchOrder(matcher, (skillopt_prompts.MERGE_FAILURE_SYSTEM,))
    b_patches = [dict(p1), dict(p2)]
    assert order(6, "failure", b_patches) == [b_patches[1], b_patches[0]]
    other = [{"reasoning": "x", "edits": []}, dict(p1)]
    assert order(6, "failure", other) == other
    assert [e["applied"] for e in order.applied] == [True, False]


def test_unused_recordings_after_b_last_served_call(tmp_path):
    """Recordings A made after the last one B was served are counted apart."""
    matcher = CallMatcher(_calls_with_rc(tmp_path, [
        ("s", "early unused", "x", 0), ("s", "served", "ok", 0), ("s", "later", "y", 0),
        ("s", "later 2", "z", 0),
    ]))
    assert matcher.unused_after_last_served() == 4
    matcher.lookup("s", "served")
    assert matcher.unused_after_last_served() == 2


def test_replay_runs_without_the_loops_holdout_audit():
    """SkillOpt has no post-search holdout audit; the replay's Loop must not
    spend model calls on one (refactory_dsh/loop.py ``audit_holdout``)."""
    import inspect

    from refactory_packages.skillopt import replay_runner

    assert "holdout=0.0" in inspect.getsource(replay_runner.run_replay)


def test_b_patch_files_list_every_group_that_returned_a_patch():
    """A lists a file for a reply with a patch even when it had no edits."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    trace = [
        _trace_line("analyst_failure", 1, group_index=0, n_edits=2, returned_patch=True),
        _trace_line("analyst_failure", 1, group_index=1, n_edits=0, returned_patch=True),
        _trace_line("analyst_success", 1, group_index=0, n_edits=0, returned_patch=False),
    ]
    entry = b_history_from_trace(trace)[0]
    assert entry["patch_files"] == ["minibatch_fail_000.json", "minibatch_fail_001.json"]
    assert entry["n_patches"] == 1


def test_b_edit_apply_report_of_an_unchanged_patch():
    """A writes edit_apply_report.json for a patch that changed nothing too."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    report = {"outcomes": [{"index": 0, "status": "skipped_not_found"}]}
    entry = b_history_from_trace([
        _trace_line("analyst_failure", 1, group_index=0, n_edits=1, returned_patch=True),
        _trace_line("patch_noop", 1, reasons=["E-noop"], n_edits=1, edit_report=report),
    ])[0]
    assert entry["edit_apply_summary"] == {"total": 1, "applied": 0, "skipped": 1, "errors": 0}


# ── the verdict: completed only when B reproduced A's run ──────────────────


def _verdict_for(history: list[dict], a_history: list[dict], tmp_path: Path,
                 unused_before: list[dict] | None = None) -> dict:
    from refactory_packages.skillopt.replay_runner import apply_verdict

    a_out = _a_out_from_b_history(tmp_path, a_history, num_epochs=3)
    report = {"replay_status": "completed", "comparison": compare_runs(a_out, history)}
    apply_verdict(report, unused_before or [])
    return report


def test_verdict_completed_when_b_reproduces_a(tmp_path):
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    report = _verdict_for(history, history, tmp_path)
    assert report["replay_status"] == "completed"
    assert "divergence" not in report


def test_verdict_diverged_when_a_post_training_score_differs(tmp_path):
    """B's post-training scores (its strategy report) must equal A's summary.json
    (trainer.py:2480-2515); one that differs makes the replay diverge, naming it."""
    from refactory_packages.skillopt.replay_runner import (
        apply_verdict, b_history_from_trace, compare_test_eval,
    )

    history = b_history_from_trace(_known_trace())
    a_out = _a_out_from_b_history(tmp_path, history, num_epochs=3)
    (a_out / "summary.json").write_text(json.dumps({
        "final_selection_hard": 0.5, "baseline_test_hard": 0.25, "test_hard": 0.75,
        "final_test_hard": 0.5}))
    same = {"test_eval": "done", "final_selection_hard": 0.5, "baseline_test_hard": 0.25,
            "test_hard": 0.75, "final_test_hard": 0.5}
    for b, status in ((same, "completed"), ({**same, "test_hard": 0.7}, "diverged"),
                      ({**same, "final_test_hard": None}, "diverged")):
        report = {"replay_status": "completed", "comparison": compare_runs(a_out, history),
                  "test_eval": compare_test_eval(a_out, b)}
        apply_verdict(report, [])
        assert report["replay_status"] == status, b
    assert report["divergence"] == {"phase": "test_eval", "field": "final_test_hard",
                                    "a_value": 0.5, "b_value": None}


def test_verdict_diverged_when_a_decision_differs(tmp_path):
    """The reviewer's plant: the no-op patch rule reverted, so B refuses where A
    gated and rejected. Every call can still match; the decisions do not."""
    import copy

    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    a_history = b_history_from_trace(_known_trace())
    b_history = copy.deepcopy(a_history)
    b_history[0]["action"] = "patch_refused"
    report = _verdict_for(b_history, a_history, tmp_path)
    assert report["replay_status"] == "diverged"
    assert (report["divergence"]["field"], report["divergence"]["a_value"],
            report["divergence"]["b_value"]) == ("accept_reject", "accept_new_best", "patch_refused")


def test_verdict_diverged_when_b_misses_an_epoch_a_ran(tmp_path):
    """A B epoch that is missing is not matched, however empty its fields."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    a_history = b_history_from_trace(_known_trace())
    report = _verdict_for(a_history[:1], a_history, tmp_path)
    summary = report["comparison"]["summary"]
    assert (summary["matched_epochs"], summary["missing_b_epochs"]) == (1, 1)
    assert report["replay_status"] == "diverged"
    assert report["divergence"]["phase"] == "epoch_missing"


def test_verdict_diverged_when_a_field_a_has_is_missing_in_b(tmp_path):
    """A has data B lacks: not a match (compare._has_data)."""
    import copy

    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    a_history = b_history_from_trace(_known_trace())
    b_history = copy.deepcopy(a_history)
    b_history[0]["merged_patch"] = None
    b_history[0]["n_edits_merged"] = None
    report = _verdict_for(b_history, a_history, tmp_path)
    assert report["comparison"]["summary"]["matched_epochs"] == 1
    assert report["replay_status"] == "diverged"
    assert report["divergence"]["epoch"] == 1


def test_verdict_diverged_when_only_a_early_stopped(tmp_path):
    """A ended before num_epochs; a B that did not stop there diverges."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    a_history = b_history_from_trace(_known_trace())
    b_history = b_history_from_trace([
        entry for entry in _known_trace() if entry.get("step_type") != "early_stop"
    ])
    report = _verdict_for(b_history, a_history, tmp_path)
    assert report["replay_status"] == "diverged"
    assert report["divergence"]["field"] == "early_stop"


def test_verdict_diverged_on_a_recording_left_before_b_last_served_call(tmp_path):
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    history = b_history_from_trace(_known_trace())
    skipped = [{"line": 7, "system": "s", "prompt": "p"}]
    report = _verdict_for(history, history, tmp_path, unused_before=skipped)
    assert report["replay_status"] == "diverged"
    assert report["divergence"]["field"] == "unused_before_b_last_served_call"
    assert report["unused_before_b_last_served_call"] == 1


def test_unused_before_last_served_lists_skipped_calls(tmp_path):
    matcher = CallMatcher(_calls_with_rc(tmp_path, [
        ("s", "skipped", "x", 0), ("s", "served", "ok", 0), ("s", "after", "y", 0),
    ]))
    matcher.lookup("s", "served")
    assert [u["line"] for u in matcher.unused_before_last_served()] == [0]


def test_slow_update_and_meta_skill_compare_their_content(tmp_path):
    """slow_result.json / meta_skill_result.json are compared on every key both
    runs record, not only ``action``: a changed guidance diverges."""
    import copy

    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    trace = _known_trace()
    for entry in trace:
        if entry.get("step_type") == "slow_update":
            entry.update(reasoning="why", slow_update_content="GUIDE", prev_hard=0.5,
                         curr_hard=0.75, candidate_hash="0123456789abcdef")
        if entry.get("step_type") == "meta_skill" and entry["epoch"] == 2:
            entry.update(reasoning="r", meta_skill_content="META")
    a_history = b_history_from_trace(trace)
    assert a_history[1]["slow_update"] == {
        "action": "force_accept", "reasoning": "why", "slow_update_content": "GUIDE",
        "prev_hard": 0.5, "curr_hard": 0.75, "candidate_hash": "0123456789abcdef",
    }
    assert a_history[1]["meta_skill"] == {
        "action": "write_meta_skill", "reasoning": "r", "meta_skill_content": "META",
    }
    assert _verdict_for(a_history, a_history, tmp_path)["replay_status"] == "completed"
    for field, key, value in (
        ("slow_update", "slow_update_content", "GUIDE plus an appended line"),
        ("slow_update", "curr_hard", 0.7),
        ("slow_update", "candidate_hash", "fedcba9876543210"),
        ("meta_skill", "meta_skill_content", "META2"),
    ):
        b_history = copy.deepcopy(a_history)
        b_history[1][field][key] = value
        report = _verdict_for(b_history, a_history, tmp_path / f"{field}-{key}")
        assert report["replay_status"] == "diverged", (field, key)
        assert report["divergence"]["field"] == field


def test_accept_is_a_new_best_only_when_it_beats_the_best(tmp_path):
    """A's ``accept_new_best`` needs a strictly higher score."""
    from refactory_packages.skillopt.replay_runner import b_history_from_trace

    entry = b_history_from_trace([
        _trace_line("initial_val", 0, score=0.5),
        _trace_line("val_gate_accept", 1, cand_score=0.5, current_score=0.4),
        _trace_line("val_gate_accept", 2, cand_score=0.6, current_score=0.5),
    ])
    assert [e["action"] for e in entry] == ["accept", "accept_new_best"]


def test_exhausted_key_sticks_as_the_first_divergence(tmp_path):
    """A key asked for more often than A recorded it ends the replay there."""
    matcher = CallMatcher(_calls_with_rc(tmp_path, [("s", "p", "ok", 0), ("s", "q", "ok", 0)]))
    matcher.lookup("s", "p")
    with pytest.raises(ReplayExhausted):
        matcher.lookup("s", "p")
    assert isinstance(matcher.first_divergence, ReplayExhausted)
    with pytest.raises(ReplayMismatch, match="already diverged"):
        matcher.lookup("s", "q")
    assert isinstance(matcher.first_divergence, ReplayExhausted)


def test_recording_matcher_marks_a_late_answers(tmp_path):
    """run_replay's matcher withholds A's late answers (recording_matcher)."""
    from refactory_dsh.model import CompletionTimeout
    from refactory_packages.skillopt.replay_runner import recording_matcher

    recording = tmp_path / "rec"
    recording.mkdir()
    _calls_with_rc(recording, [("SYS", "USER", "<answer>D</answer>", 0)])
    rollout = recording / "out" / "steps" / "step_0001" / "rollout"
    item = rollout / "predictions" / "item1"
    item.mkdir(parents=True)
    (item / "target_system_prompt.txt").write_text("SYS")
    (item / "target_user_prompt.txt").write_text("USER")
    (item / "conversation.json").write_text(json.dumps(
        [{"type": "message", "turn": 1, "content": "<answer>D</answer>"}]
    ))
    (rollout / "results.jsonl").write_text(
        json.dumps({"id": "item1", "fail_reason": "task-timeout-1860s"}) + "\n"
    )
    matcher, events = recording_matcher(recording)
    assert [e["line"] for e in events["events"]] == [0]
    with pytest.raises(CompletionTimeout):
        matcher.lookup("SYS", "USER", attempts=5)


def test_run_replay_reports_a_fault_in_the_replay_client(tmp_path, monkeypatch, _arm_b_recording):
    """B carries on past a failed optimizer call; the runner must not: a fault in
    the replay itself makes the replay an ``error``."""
    from refactory_packages.skillopt import replay as replay_mod
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)
    original = replay_mod.CallMatcher.lookup

    def lookup(self, system, prompt, *, attempts=None):  # noqa: ANN001
        if current_call_key() is not None:  # an optimizer call
            raise TypeError("bug in the replay")
        return original(self, system, prompt, attempts=attempts)

    monkeypatch.setattr(replay_mod.CallMatcher, "lookup", lookup)
    report = run_replay(recording_dir, tmp_path / "out", split_dir=LIVEMATH_SPLIT,
                        data_path=LIVEMATH_FIXTURES)
    assert report["replay_status"] == "error"
    assert report["harness_error"] == "TypeError: bug in the replay"


def test_run_replay_diverges_when_b_stops_early(tmp_path, monkeypatch, _arm_b_recording):
    """The reviewer's plant: B's early stop fires an epoch too soon. Every call B
    made matched, but B never reached A's last epoch: ``diverged``, exit 1."""
    import sys as _sys

    from refactory_packages.skillopt import replay_runner
    from refactory_packages.skillopt.loop import SkillOptLoop

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    recordings, b_history = _arm_b_recording
    recording_dir = _recording_dir(tmp_path, recordings, b_history)

    def stop_now(self, run):  # noqa: ANN001
        self._trace.record_decision("early_stop", reason="patience")
        run.stop("patience_exhausted")
        return True

    monkeypatch.setattr(SkillOptLoop, "_early_stop", stop_now)
    out = tmp_path / "out"
    monkeypatch.setattr(_sys, "argv", [
        "replay_runner", "--recording-dir", str(recording_dir), "--split-dir",
        str(LIVEMATH_SPLIT), "--data-path", str(LIVEMATH_FIXTURES), "--output-dir", str(out),
    ])
    with pytest.raises(SystemExit) as exit_info:
        replay_runner.main()
    assert exit_info.value.code == 1
    report = json.loads((out / "replay_report.json").read_text())
    # B, stopped early, goes on to its post-training evaluation of skills A never
    # tested, so its first unmatched solver call may end the replay before the
    # step comparison does; either way the replay fails and the comparison names
    # the epochs B never reached.
    assert report["replay_status"] in ("diverged", "mismatch")
    if report["replay_status"] == "diverged":
        assert report["divergence"]["field"] in ("early_stop", "epoch_reached")
    assert report["comparison"]["summary"]["diverged_epochs"] >= 1


def test_client_takes_the_claude_backend_framing_off_solver_prompts_only(tmp_path):
    """The solve node sends the question framed as A's Claude backend framed it
    (livemath ops.claude_backend_prompt); A's recording of that exact prompt is
    indexed unframed, so the solver call matches it. An optimizer call is matched
    exactly as B wrote it."""
    from refactory_packages.livemath.ops import claude_backend_prompt

    question = "## Question\nQ?\n  indented\n\n## Choices\nA. x\nB. y"
    client = ReplayClient(CallMatcher(_make_calls_jsonl(tmp_path, [
        ("solver_sys", question, "<answer>A</answer>"),
        ("opt_sys", question, "patch"),
    ])))
    assert client.complete(claude_backend_prompt(question), system="solver_sys") == "<answer>A</answer>"
    with pytest.raises(ReplayMismatch), optimizer_call(("rank", 1, 0, 0)):
        client.complete(claude_backend_prompt(question), system="opt_sys")


# ── claude_backend_wrapper: arm A's framing of optimizer calls ────────────

#: The optimizer steps arm A sends through chat_optimizer (claude_backend.py:327-329).
_OPTIMIZER_STEPS = ("analyst_failure", "analyst_success", "merge_failure", "merge_success",
                    "merge_final", "rank", "slow_update", "meta_skill")


class _RankingRecordingClient(_RecordingClient):
    """Merges return more edits than the budget, so the epoch also ranks them."""

    @staticmethod
    def _reply(step_type: str) -> str:
        if step_type.startswith("merge_"):
            return json.dumps({"reasoning": "merged", "edits": [
                {"op": "append", "content": f"## Merged {n}\nCombined."} for n in range(3)
            ]})
        return _RecordingClient._reply(step_type)


@pytest.fixture(scope="module")
def _arm_b_framed(tmp_path_factory):
    """Arm B recorded with claude_backend_wrapper off and on."""
    off, on = _RankingRecordingClient(), _RankingRecordingClient()
    _record_arm_b(tmp_path_factory.mktemp("off"), client=off)
    recordings, b_history = _record_arm_b(
        tmp_path_factory.mktemp("on"), claude_backend_wrapper=True, client=on)
    return off, on, recordings, b_history


def test_claude_backend_wrapper_frames_every_optimizer_call(_arm_b_framed):
    """On, every optimizer call is the bare message framed by claude_backend_prompt;
    off (the default), every optimizer prompt is the bare message, as before.
    Solver calls are the same either way."""
    from refactory_packages.skillopt.claude_backend import claude_backend_prompt

    off, on, _, _ = _arm_b_framed
    off_opt = sorted((step, prompt) for step, prompt in off.sent if step)
    on_opt = sorted((step, prompt) for step, prompt in on.sent if step)
    steps = {step.rsplit("_", 1)[0] if step.startswith("analyst_") else step for step, _ in off_opt}
    assert {"analyst", "merge_final", "rank", "slow_update", "meta_skill"} <= steps, steps
    assert len(off_opt) == len(on_opt) > 0
    assert not any(prompt.startswith("History:") for _, prompt in off_opt)
    assert on_opt == sorted((step, claude_backend_prompt(prompt)) for step, prompt in off_opt)
    assert sorted(p for s, p in off.sent if not s) == sorted(p for s, p in on.sent if not s)


def test_run_replay_matches_with_claude_backend_wrapper(tmp_path, monkeypatch, _arm_b_framed):
    """With the option on, the replay takes the framing off optimizer prompts too and
    matches every recorded call; the option is passed to arm B."""
    from refactory_packages.skillopt.replay_runner import run_replay

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(LIVEMATH_FIXTURES))
    _, _, recordings, b_history = _arm_b_framed
    recording_dir = _recording_dir(tmp_path, recordings, b_history)
    report = run_replay(recording_dir, tmp_path / "out", split_dir=LIVEMATH_SPLIT,
                        data_path=LIVEMATH_FIXTURES, claude_backend_wrapper=True)
    assert report["replay_status"] == "completed", report["mismatch"]
    assert report["strategy_options"]["claude_backend_wrapper"] is True
    assert report["calls_matched"] == report["calls_total"] > 0
    assert report["comparison"]["first_divergence"] is None


def test_replay_client_unframes_optimizer_prompts_only_when_asked(tmp_path):
    from refactory_packages.skillopt.claude_backend import claude_backend_prompt

    calls = _make_calls_jsonl(tmp_path, [("opt", "patch me\n  please", "r")])
    with optimizer_call(("rank", 1, 0, 0)):
        framed = claude_backend_prompt("patch me\n  please")
        assert ReplayClient(CallMatcher(calls), unframe_optimizer=True).complete(
            framed, system="opt") == "r"
        with pytest.raises(ReplayMismatch):
            ReplayClient(CallMatcher(calls)).complete(framed, system="opt")
