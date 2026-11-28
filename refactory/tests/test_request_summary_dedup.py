"""Tests for the request-summary, sharper analyst findings, trace dedup, and
propose-only replay features.

Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from refactory_dsh.optim import analysts, base
from refactory_dsh.optim.base import ProposalContext
from refactory_dsh.optim import request_summary


def _rec(item: str, passed: bool, *, steps: dict[str, tuple[str, str]] | None = None,
         meta: dict[str, str] | None = None,
         missing_steps: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = steps or {"draft": (f"DRAFT-IN {item}", f"DRAFT-OUT {item}"),
                      "final": (f"FINAL-IN {item}", f"FINAL-OUT {item}")}
    return {"id": item, "passed": passed, "score": float(passed), "repeat": 1,
            "prediction": f"PRED {item}", "label": f"GOLD {item}",
            "fail_reason": None if passed else f"GRADER {item}",
            "error_text": None,
            "metadata": {"question": f"QUESTION {item}"},
            "item_meta": meta or {},
            "missing_steps": missing_steps or {},
            "exchanges": [{"node": node, "prompt": given, "response": out}
                          for node, (given, out) in steps.items()]}


def _ctx(records: list[dict[str, Any]], **overrides: Any) -> ProposalContext:
    return ProposalContext(
        mode_name="m", epoch=3, item_records=records,
        analysts=overrides.pop("analysts", {}),
        nodes=[{"id": "draft", "kind": "agent"}, {"id": "final", "kind": "agent"}],
        edges=[{"source": "draft", "target": "final"}],
        node_states=[{"id": "draft", "kind": "agent", "prompt_template": "TEMPLATE-D",
                      "knobs": ["draft__task"],
                      "field_bindings": [["prompt_template", "draft__task"]]},
                     {"id": "final", "kind": "agent", "prompt_template": "TEMPLATE-F"}],
        knobs=[{"name": "draft__task", "kind": "str", "value": "BOUND-VALUE", "frozen": False},
               {"name": "style", "kind": "str", "value": "STYLE-VALUE", "frozen": False}],
        **overrides)


# ── A: Request summary tables ────────────────────────────────────────────


def test_build_noise_with_repeated_scorings():
    records_by_epoch = {
        "c1": {1: [{"score": 0.6}, {"score": 0.7}],
               2: [{"score": 0.5}, {"score": 0.5}]},
        "c2": {1: [{"score": 0.8}]},
    }
    result = request_summary.build_noise(records_by_epoch, {})
    assert result is not None
    assert len(result["spreads"]) == 1, "only c1 has ≥ 2 epochs"
    spread = result["spreads"][0]
    assert spread["candidate"] == "c1"
    assert spread["epochs"] == 2
    assert spread["spread"] == round(0.65 - 0.5, 4)


def test_build_noise_returns_none_without_repeats():
    records_by_epoch = {"c1": {1: [{"score": 0.5}]}}
    assert request_summary.build_noise(records_by_epoch, {}) is None


def test_build_item_stability_always_never_sometimes():
    train = {
        "c1": [_rec("a", True), _rec("b", False), _rec("c", True)],
        "c2": [_rec("a", True), _rec("b", False), _rec("c", False)],
    }
    result = request_summary.build_item_stability(train)
    assert result is not None
    assert result["always_pass"] == 1, "item a passes in both"
    assert result["never_pass"] == 1, "item b fails in both"
    assert len(result["sometimes_pass"]) == 1, "item c flips"
    assert result["sometimes_pass"][0]["item"] == "c"
    assert result["sometimes_pass"][0]["pass_rate"] == 0.5
    assert result["scorings"] == 2


def test_build_item_stability_returns_none_without_records():
    assert request_summary.build_item_stability({}) is None


def test_build_by_group_discovers_categorical_field():
    records = [
        _rec("t1", True, meta={"topic": "math"}),
        _rec("t2", False, meta={"topic": "math"}),
        _rec("t3", True, meta={"topic": "science"}),
        _rec("t4", False, meta={"topic": "science"}),
    ]
    result = request_summary.build_by_group(records, None)
    assert result is not None
    assert result["field"] == "topic"
    assert len(result["groups"]) == 2
    names = {g["group"] for g in result["groups"]}
    assert names == {"math", "science"}


def test_build_by_group_returns_none_without_categorical_field():
    records = [_rec("t1", True), _rec("t2", False)]
    assert request_summary.build_by_group(records, None) is None


def test_build_by_group_with_best_candidate():
    edited = [
        _rec("t1", False, meta={"topic": "math"}),
        _rec("t2", False, meta={"topic": "science"}),
    ]
    best = [
        _rec("t1", True, meta={"topic": "math"}),
        _rec("t2", True, meta={"topic": "science"}),
    ]
    result = request_summary.build_by_group(edited, best)
    assert result is not None
    for g in result["groups"]:
        assert g["edited"] == 0.0
        assert g["best"] == 1.0


def test_build_by_step_from_missing_steps():
    records = [
        _rec("t1", False, missing_steps={"draft": {"cause": "timeout"}}),
        _rec("t2", False, missing_steps={"draft": {"cause": "error"}}),
        _rec("t3", False, missing_steps={"final": {"cause": "timeout"}}),
    ]
    result = request_summary.build_by_step(records, None)
    assert result is not None
    steps = {s["step"]: s for s in result["steps"]}
    assert "draft" in steps
    assert steps["draft"]["no_answer"] == 2
    assert "final" in steps
    assert steps["final"]["no_answer"] == 1


def test_build_by_step_from_findings():
    findings = [
        {"steps": {"first_wrong": "draft"}, "items": ["t1", "t2"]},
        {"steps": {"first_wrong": "final"}, "items": ["t3"]},
    ]
    result = request_summary.build_by_step([], findings)
    assert result is not None
    steps = {s["step"]: s for s in result["steps"]}
    assert steps["draft"]["first_wrong"] == 2
    assert steps["final"]["first_wrong"] == 1


def test_build_by_step_returns_none_without_data():
    assert request_summary.build_by_step([], None) is None


def test_build_change_history_classifies_edits():
    history = [
        {"edits": [], "score": 0.6, "parent_score": 0.5},
        {"edits": [], "score": 0.4, "parent_score": 0.5},
        {"edits": [{"op": "insert_after", "node": {"kind": "agent"}}],
         "score": 0.7, "parent_score": 0.5},
    ]
    result = request_summary.build_change_history(history)
    assert result is not None
    kinds = {r["kind"]: r for r in result}
    assert "other" in kinds
    assert kinds["other"]["count"] == 2
    assert kinds["other"]["mean_change"] == pytest.approx(0.0)
    assert kinds["other"]["best"] == pytest.approx(0.1)
    assert kinds["other"]["worst"] == pytest.approx(-0.1)


def test_build_change_history_returns_none_without_data():
    assert request_summary.build_change_history([]) is None


def test_section_lines_has_all_headings_when_data_present():
    summary = {
        "noise": {"spreads": [{"candidate": "c1", "epochs": 2, "spread": 0.1}]},
        "item_stability": {"always_pass": 1, "never_pass": 1, "sometimes_pass": [],
                           "scorings": 2},
        "by_group": {"field": "topic", "groups": [
            {"group": "math", "items": 2, "edited": 0.5, "best": 0.8}]},
        "by_step": {"steps": [{"step": "draft", "first_wrong": 2, "no_answer": 1}]},
        "change_history": [{"kind": "other", "count": 1, "mean_change": 0.1,
                            "best": 0.1, "worst": 0.1}],
    }
    lines = request_summary.section_lines(summary)
    text = "\n".join(lines)
    assert "## Computed summary" in text
    assert "### Noise" in text
    assert "### Item stability" in text
    assert "### By group" in text
    assert "### By step" in text
    assert "### History by kind of change" in text


def test_section_lines_empty_without_data():
    assert request_summary.section_lines(None) == []
    assert request_summary.section_lines({}) == []


# ── B: Sharper analyst findings ──────────────────────────────────────────


def test_finding_captures_can_step_fix_it_when_present():
    raw = {"finding": "step draft fails at X", "where": {"nodes": ["draft"]},
           "items": ["t1"], "impact": "high",
           "can_step_fix_it": "no, needs info from earlier step"}
    result = analysts.finding(raw, "failure", {"t1"})
    assert result is not None
    assert result["can_step_fix_it"] == "no, needs info from earlier step"


def test_finding_omits_can_step_fix_it_when_absent():
    raw = {"finding": "step draft fails at X", "where": {"nodes": ["draft"]},
           "items": ["t1"], "impact": "high"}
    result = analysts.finding(raw, "failure", {"t1"})
    assert result is not None
    assert "can_step_fix_it" not in result


def test_finding_omits_can_step_fix_it_when_empty():
    raw = {"finding": "step draft fails at X", "where": {"nodes": ["draft"]},
           "items": ["t1"], "impact": "high", "can_step_fix_it": ""}
    result = analysts.finding(raw, "failure", {"t1"})
    assert result is not None
    assert "can_step_fix_it" not in result


def test_failure_answer_format_includes_can_step_fix_it():
    assert "can_step_fix_it" in analysts.FAILURE_ANSWER


def test_success_answer_format_includes_also_in_failures():
    assert "also_in_failures" in analysts.SUCCESS_ANSWER


def test_contrast_system_prompt_exists():
    assert "contrast" in analysts.CONTRAST_SYSTEM.lower()
    assert "PASSED" in analysts.CONTRAST_SYSTEM
    assert "FAILED" in analysts.CONTRAST_SYSTEM


def test_contrast_batch_renders_passing_and_failing_traces():
    pass_rec = _rec("t1", True)
    fail_rec = _rec("t1", False)
    flip_items = [{"item": "t1", "pass_rate": 0.5, "scorings": 2,
                   "pass_record": pass_rec, "fail_record": fail_rec}]
    ctx = _ctx([pass_rec, fail_rec])
    pipeline = "step draft -> step final"
    result = analysts.contrast_batch(ctx, flip_items, pipeline)
    assert result is not None
    prompt, clip = result
    assert "**Passing run:**" in prompt
    assert "**Failing run:**" in prompt
    assert "DRAFT-OUT t1" in prompt
    assert "t1" in prompt


def test_contrast_batch_returns_none_without_flip_items():
    ctx = _ctx([_rec("t1", False)])
    assert analysts.contrast_batch(ctx, [], "pipeline") is None


def test_contrast_batch_includes_multiple_items():
    items = []
    flip_items = []
    for i in range(3):
        item_id = f"flip{i}"
        pr = _rec(item_id, True)
        fr = _rec(item_id, False)
        items += [pr, fr]
        flip_items.append({"item": item_id, "pass_rate": 0.5, "scorings": 2,
                           "pass_record": pr, "fail_record": fr})
    ctx = _ctx(items)
    prompt, _clip = analysts.contrast_batch(ctx, flip_items, "step draft -> step final")
    for fi in flip_items:
        assert f"### Item `{fi['item']}`" in prompt


def test_mark_targeted_by_node_overlap():
    findings = [{"finding": "draft step fails badly", "where": {"nodes": ["step_a"]},
                 "items": ["t1"]}]
    targets = [{"round": 2, "targets": "fix step_a", "change": "+0.05",
                "edits": [{"node": "step_a"}]}]
    result = analysts._mark_targeted(findings, targets)
    assert result[0].get("already_targeted") == "round 2; score change: +0.05"


def test_mark_targeted_by_knob_overlap():
    findings = [{"finding": "prompt is too vague", "where": {"knobs": ["step_a__task"]},
                 "items": ["t1"]}]
    targets = [{"round": 1, "targets": "", "change": "+0.02",
                "edits": [{"name": "step_a__task"}]}]
    result = analysts._mark_targeted(findings, targets)
    assert "already_targeted" in result[0]
    assert "round 1" in result[0]["already_targeted"]


def test_mark_targeted_no_match():
    findings = [{"finding": "draft step fails badly", "where": {"nodes": ["step_a"]},
                 "items": ["t1"]}]
    targets = [{"round": 2, "targets": "unrelated thing", "change": "",
                "edits": [{"node": "step_z"}]}]
    result = analysts._mark_targeted(findings, targets)
    assert "already_targeted" not in result[0]


def test_mark_targeted_empty_targets():
    findings = [{"finding": "x", "where": {"nodes": ["a"]}, "items": []}]
    result = analysts._mark_targeted(findings, [])
    assert "already_targeted" not in result[0]


def test_finding_line_shows_already_targeted():
    entry = {"source": "failure", "finding": "draft step fails badly",
             "where": {"nodes": ["draft"]}, "items": ["t1"], "impact": "high",
             "already_targeted": "round 2; score change: +0.05"}
    text = analysts.finding_line(1, entry)
    assert "ALREADY TARGETED" in text
    assert "round 2" in text
    assert "+0.05" in text


def test_finding_line_shows_can_step_fix_it():
    entry = {"source": "failure", "finding": "draft step fails",
             "where": {"nodes": ["draft"]}, "items": ["t1"], "impact": "high",
             "can_step_fix_it": "no, needs external info"}
    text = analysts.finding_line(1, entry)
    assert "can the step fix it" in text
    assert "no, needs external info" in text


def test_section_lines_mentions_contrast_batch():
    analysis = {
        "batches": [
            {"kind": "failure", "items": ["t1"], "index": 1},
            {"kind": "success", "items": ["t2"], "index": 1},
            {"kind": "contrast", "items": ["t3", "t4"], "index": 1},
        ],
        "findings": [{"source": "failure", "finding": "x", "where": {},
                       "items": ["t1"], "impact": "high"}],
        "group_size": 8,
    }
    lines = analysts.section_lines(analysis)
    text = "\n".join(lines)
    assert "contrast batch" in text
    assert "2 item(s) compared" in text


def test_section_lines_passes_proposal_targets_to_mark_targeted():
    analysis = {
        "batches": [{"kind": "failure", "items": ["t1"], "index": 1}],
        "findings": [{"source": "failure", "finding": "long finding about step_a",
                       "where": {"nodes": ["step_a"]}, "items": ["t1"], "impact": "high"}],
        "group_size": 8,
    }
    targets = [{"round": 1, "targets": "", "change": "+0.1",
                "edits": [{"node": "step_a"}]}]
    lines = analysts.section_lines(analysis, targets)
    text = "\n".join(lines)
    assert "ALREADY TARGETED" in text


# ── C: Deduplicate traces ────────────────────────────────────────────────


def test_no_inline_full_traces_when_analysts_ran():
    records = [_rec(f"t{i}", i % 2 == 0) for i in range(20)]
    ctx = _ctx(records, analysis={"batches": [], "findings": []})
    rendered = ctx.render()
    assert "Every item" in rendered, "the table must still appear"
    assert "Some items in full" not in rendered, "inline traces must be omitted"


def test_every_item_table_has_group_and_first_wrong_columns_when_analysts_ran():
    records = [_rec("t1", False, meta={"topic": "math"}),
               _rec("t2", True, meta={"topic": "science"})]
    findings = [{"source": "failure", "finding": "x",
                 "where": {"nodes": ["draft"]},
                 "steps": {"first_wrong": "draft"},
                 "items": ["t1"], "impact": "high"}]
    ctx = _ctx(records, analysis={"batches": [], "findings": findings})
    rendered = ctx.render()
    assert "group" in rendered.lower()
    assert "first wrong step" in rendered.lower()


def test_need_traces_in_answer_format_when_analysts_ran():
    text = base.answer_instructions([], analysts_ran=True)
    assert "need_traces" in text


def test_need_traces_absent_from_answer_format_without_analysts():
    text = base.answer_instructions([], analysts_ran=False)
    assert "need_traces" not in text


def test_render_traces_for_items_shows_requested_item():
    records = [_rec("t1", False), _rec("t2", True)]
    result = base.render_traces_for_items(["t1"], records)
    assert "t1" in result
    assert "DRAFT-OUT t1" in result
    assert "1 requested item" in result


def test_render_traces_for_items_handles_missing_item():
    records = [_rec("t1", False)]
    result = base.render_traces_for_items(["t999"], records)
    assert "not found" in result


def test_render_traces_for_items_includes_answer_instruction():
    records = [_rec("t1", False)]
    result = base.render_traces_for_items(["t1"], records)
    assert "final answer" in result.lower()


def test_inline_traces_shown_when_analysts_did_not_run():
    records = [_rec("t1", False), _rec("t2", True)]
    ctx = _ctx(records, analysis=None)
    rendered = ctx.render()
    assert "DRAFT-OUT t1" in rendered, "traces should be inline without analysts"


# ── D: Val/test content guard ────────────────────────────────────────────


def test_assert_no_leakage_raises_on_val_content():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from propose_only import _assert_no_leakage

    tokens = {"This is secret validation content that must not leak"}
    with pytest.raises(AssertionError, match="val/test content leaked"):
        _assert_no_leakage(
            "Here is a request with This is secret validation content that must not leak in it",
            tokens, "test request")


def test_assert_no_leakage_passes_when_clean():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from propose_only import _assert_no_leakage

    tokens = {"This is secret validation content that must not leak"}
    _assert_no_leakage("Here is a clean request with no secrets", tokens, "test request")


def test_load_val_test_content_reads_val_items(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from propose_only import _load_val_test_content

    val_items = [{"input": "A very long validation input text that should be detected",
                  "expected": "short"}]
    (tmp_path / "val_items.json").write_text(json.dumps(val_items))
    tokens = _load_val_test_content(tmp_path)
    assert "A very long validation input text that should be detected" in tokens
    assert "short" not in tokens, "strings ≤ 20 chars are not collected"


# ── Mutation checks ──────────────────────────────────────────────────────


def test_summary_section_nonempty_when_data_present():
    summary = {"noise": {"spreads": [{"candidate": "c1", "epochs": 2, "spread": 0.05}]}}
    lines = request_summary.section_lines(summary)
    assert len(lines) > 1, "reverting section_lines to [] breaks this"


def test_contrast_prompt_has_passing_and_failing_labels():
    pr = _rec("flip1", True)
    fr = _rec("flip1", False)
    ctx = _ctx([pr, fr])
    prompt, _ = analysts.contrast_batch(
        ctx, [{"item": "flip1", "pass_rate": 0.5, "scorings": 2,
               "pass_record": pr, "fail_record": fr}],
        "step draft -> step final")
    assert "**Passing run:**" in prompt, "reverting contrast_batch breaks this"
    assert "**Failing run:**" in prompt, "reverting contrast_batch breaks this"


def test_already_targeted_label_appears_in_finding_line():
    entry = {"source": "failure", "finding": "some issue",
             "where": {"nodes": ["draft"]}, "items": ["t1"],
             "already_targeted": "round 3"}
    line = analysts.finding_line(1, entry)
    assert "ALREADY TARGETED" in line, "reverting finding_line removes the annotation"
    assert "round 3" in line


def test_can_step_fix_it_in_finding_round_trips():
    raw = {"finding": "fail", "where": {}, "items": ["t1"], "impact": "high",
           "can_step_fix_it": "yes"}
    parsed = analysts.finding(raw, "failure", {"t1"})
    assert parsed["can_step_fix_it"] == "yes", "reverting finding() drops can_step_fix_it"
    line = analysts.finding_line(1, parsed)
    assert "can the step fix it" in line, "reverting finding_line drops it"
