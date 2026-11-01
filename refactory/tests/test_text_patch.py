"""Tests for the text_patch edit primitive (F7).

Each test fails when the feature it tests is reverted.
"""

from __future__ import annotations

import copy
import json

import pytest

from refactory_dsh.nn import edits
from refactory_dsh.nn.edits import EditError, SKILLOPT_PATCH_CONFIG, apply
from refactory_dsh.optim.base import Candidate, Proposal


# ── helpers ────────────────────────────────────────────────────────────────

def _state(text: str = "hello world", *, frozen: bool = False, kind: str = "str") -> dict:
    """Minimal state with a str knob and a trivial graph."""
    from refactory_dsh.nn import Graph, Node

    graph = Graph()
    graph.add(Node("start", "op", call="json:dumps", knobs=("prompt",)))
    spec: dict = {"kind": kind, "default": text if kind == "str" else 0}
    if frozen:
        spec["frozen"] = True
    if kind == "int":
        spec["lo"] = 0
        spec["hi"] = 100
    return {
        "graph": graph.to_state(),
        "knobs": {
            "prompt": {
                "value": text if kind == "str" else 0,
                "spec": spec,
            }
        },
    }


def _patch(edits_list: list[dict], *, config: dict | None = None, name: str = "prompt") -> dict:
    edit = {"op": "text_patch", "name": name, "edits": edits_list}
    if config is not None:
        edit["config"] = config
    return edit


def _apply_with_report(state: dict, edit: dict) -> tuple[dict, list]:
    """Apply and return (new_state, edit_reports)."""
    reports: list = []
    result = apply(edit, state, edit_reports=reports)
    return result, reports


def _apply(state: dict, edit: dict) -> dict:
    return apply(edit, state)


def _value(result: dict) -> str:
    return result["knobs"]["prompt"]["value"]


# ── edit kinds ─────────────────────────────────────────────────────────────

def test_replace_first_occurrence():
    """replace acts on the first occurrence only when text contains multiple matches."""
    state = _state("aaa bbb aaa")
    edit = _patch([{"kind": "replace", "target": "aaa", "content": "ccc"}])
    result = _apply(state, edit)
    assert _value(result) == "ccc bbb aaa"


def test_delete_first_occurrence():
    """delete removes only the first occurrence."""
    state = _state("foo bar foo baz")
    edit = _patch([{"kind": "delete", "target": "foo "}])
    result = _apply(state, edit)
    assert _value(result) == "bar foo baz"


def test_insert_after_end_of_line():
    """insert_after places content after the end of the line containing the target."""
    state = _state("line1 target here\nline2\nline3")
    edit = _patch([{"kind": "insert_after", "target": "target", "content": "INSERTED"}])
    result = _apply(state, edit)
    assert _value(result) == "line1 target here\nINSERTED\nline2\nline3"


def test_insert_after_last_line():
    """insert_after on the last line (no trailing newline) appends."""
    state = _state("only line with target")
    edit = _patch([{"kind": "insert_after", "target": "target", "content": "AFTER"}])
    result = _apply(state, edit)
    assert _value(result) == "only line with target\nAFTER"


def test_insert_before_start_of_line():
    """insert_before places content before the start of the line containing the target."""
    state = _state("line1\nline2 target\nline3")
    edit = _patch([{"kind": "insert_before", "target": "target", "content": "BEFORE"}])
    result = _apply(state, edit)
    assert _value(result) == "line1\nBEFORE\nline2 target\nline3"


def test_append():
    """append adds content at the end."""
    state = _state("existing content")
    edit = _patch([{"kind": "append", "content": "NEW STUFF"}])
    result = _apply(state, edit)
    assert _value(result) == "existing content\n\nNEW STUFF\n"


# ── on_missing leniency ───────────────────────────────────────────────────

def test_on_missing_skip():
    """target not found with on_missing='skip' skips with skipped_*_target_not_found."""
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "MISSING", "content": "X"}],
        config={"on_missing": "skip"},
    )
    result, reports = _apply_with_report(state, edit)
    assert _value(result) == "hello world"
    assert reports[0]["outcomes"][0]["status"] == "skipped_replace_target_not_found"


def test_on_missing_append():
    """target not found with on_missing='append' appends content."""
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "MISSING", "content": "APPENDED"}],
        config={"on_missing": "append"},
    )
    result, reports = _apply_with_report(state, edit)
    assert "APPENDED" in _value(result)
    assert reports[0]["outcomes"][0]["status"] == "applied_append"


def test_on_missing_refuse():
    """target not found with on_missing='refuse' raises EditError, state untouched."""
    state = _state("hello world")
    before = copy.deepcopy(state)
    edit = _patch(
        [{"kind": "replace", "target": "MISSING", "content": "X"}],
        config={"on_missing": "refuse"},
    )
    with pytest.raises(EditError) as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-patch"
    assert state == before


def test_on_missing_per_kind_dict():
    """on_missing as a dict: insert_after with 'append' falls back, replace with 'skip' skips."""
    state = _state("hello world")
    edit = _patch(
        [
            {"kind": "insert_after", "target": "MISSING_IA", "content": "FALLBACK"},
            {"kind": "replace", "target": "MISSING_R", "content": "NOPE"},
        ],
        config={"on_missing": {"insert_after": "append", "replace": "skip"}},
    )
    result, reports = _apply_with_report(state, edit)
    val = _value(result)
    assert "FALLBACK" in val
    assert "NOPE" not in val
    outcomes = reports[0]["outcomes"]
    assert outcomes[0]["status"] == "applied_append"
    assert outcomes[1]["status"] == "skipped_replace_target_not_found"


# ── protected regions ─────────────────────────────────────────────────────

def test_protected_region_skips_edit():
    """edit whose target falls inside a protected region is skipped."""
    text = "before <!-- PROTECTED -->secret stuff<!-- /PROTECTED --> after"
    state = _state(text)
    edit = _patch(
        [{"kind": "replace", "target": "secret stuff", "content": "CHANGED"}],
        config={
            "honour_protected": True,
            "protected_regions": [["<!-- PROTECTED -->", "<!-- /PROTECTED -->"]],
        },
    )
    result, reports = _apply_with_report(state, edit)
    assert _value(result) == text  # unchanged
    assert reports[0]["outcomes"][0]["status"] == "skipped_protected_region"


def test_marker_stripping_from_content():
    """protection markers in edit content are stripped before application."""
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "world", "content": "<!-- START -->brave new<!-- END --> world"}],
        config={
            "honour_protected": True,
            "protected_regions": [["<!-- START -->", "<!-- END -->"]],
            "strip_markers": True,
        },
    )
    result = _apply(state, edit)
    val = _value(result)
    assert "<!-- START -->" not in val
    assert "<!-- END -->" not in val
    assert "brave new world" in val


def test_append_before_protected_region():
    """With honour_protected=True and protected regions, append inserts before the earliest marker."""
    text = "content here\n<!-- PROTECTED -->do not touch<!-- /PROTECTED -->"
    state = _state(text)
    edit = _patch(
        [{"kind": "append", "content": "NEW LINE"}],
        config={
            "honour_protected": True,
            "protected_regions": [["<!-- PROTECTED -->", "<!-- /PROTECTED -->"]],
        },
    )
    result = _apply(state, edit)
    val = _value(result)
    prot_idx = val.find("<!-- PROTECTED -->")
    new_idx = val.find("NEW LINE")
    assert new_idx < prot_idx, f"NEW LINE at {new_idx} should be before protected at {prot_idx}"


# ── outcome report ────────────────────────────────────────────────────────

def test_per_edit_outcome_report():
    """multi-edit patch returns report via edit_reports collector, not edit dict."""
    state = _state("alpha beta gamma")
    edit = _patch([
        {"kind": "replace", "target": "beta", "content": "BETA"},
        {"kind": "delete", "target": "MISSING"},
        {"kind": "append", "content": "tail"},
    ])
    _result, reports = _apply_with_report(state, edit)
    report = reports[0]
    assert len(report["outcomes"]) == 3
    assert report["outcomes"][0]["index"] == 0
    assert report["outcomes"][0]["kind"] == "replace"
    assert report["outcomes"][0]["status"] == "applied_replace"
    assert report["outcomes"][1]["index"] == 1
    assert report["outcomes"][1]["status"] == "skipped_delete_target_not_found"
    assert report["outcomes"][2]["index"] == 2
    assert report["outcomes"][2]["status"] == "applied_append"


def test_outcome_report_serialisable():
    """The report dict is serialisable (for journaling)."""
    state = _state("hello world")
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    _result, reports = _apply_with_report(state, edit)
    report = reports[0]
    # Must be JSON-serialisable
    serialised = json.dumps(report)
    assert "applied_replace" in serialised
    # noop is False because text changed
    assert report["noop"] is False


def test_noop_patch():
    """patch whose edits all skip is reported as noop=True."""
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "MISSING", "content": "X"}],
        config={"on_missing": "skip"},
    )
    _result, reports = _apply_with_report(state, edit)
    assert reports[0]["noop"] is True


# ── report not in edit dict (brief fix 1) ─────────────────────────────────

def test_report_not_in_edit_dict():
    """After apply(), the edit dict has no _report key.  Fails when the report
    leak is reintroduced (edit dict mutation returns).
    """
    state = _state("hello world")
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    _apply(state, edit)
    assert "_report" not in edit


def test_report_not_in_describe():
    """A Candidate's describe() output contains no report text by default.
    Reports live in candidate.edit_reports and are omitted from describe()
    unless explicitly requested.
    """
    state = _state("hello world")
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    reports: list = []
    result = apply(edit, state, edit_reports=reports)
    candidate = Candidate.from_state(
        result,
        parent=None,
        edits=[edit],
        edit_reports=[reports[0] if reports else None],
    )
    desc = candidate.describe()
    desc_text = json.dumps(desc, default=str)
    # The default describe() must not contain report data
    assert "applied_replace" not in desc_text
    assert "outcomes" not in desc_text
    assert "noop" not in desc_text
    # But include_reports=True does expose it
    desc_with = candidate.describe(include_reports=True)
    assert "edit_reports" in desc_with
    assert desc_with["edit_reports"][0] is not None


def test_report_survives_stop_resume(tmp_path):
    """A candidate with edit_reports is journaled, then replayed from a fresh
    journal walk; the restored candidate's edit_reports match the originals.
    """
    from refactory_dsh.loop import Loop, LoopConfig

    state = _state("hello world")
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    reports: list = []
    result = apply(edit, state, edit_reports=reports)
    report = reports[0]

    config = LoopConfig(
        epochs=1, batch_size=2, optimizer="random", seed=5,
        runs_root=tmp_path,
    )
    loop = Loop("circle_packing/adaptive", config, project=tmp_path.parent)
    loop.init()

    # Manually create a candidate with edit_reports and journal it
    candidate = Candidate.from_state(
        result,
        parent=loop.population[0].id,
        edits=[edit],
        edit_reports=[report],
    )
    loop._journal_candidate(candidate)

    # Also journal the seed candidate so replay has at least one
    loop._journal_candidate(loop.population[0])

    # Now replay from scratch
    config2 = LoopConfig(
        epochs=2, batch_size=2, optimizer="random", seed=5,
        runs_root=tmp_path, resume=True,
    )
    loop2 = Loop("circle_packing/adaptive", config2, project=tmp_path.parent)
    loop2.init()

    # Find the replayed candidate
    restored = None
    for c in loop2.population:
        if c.id == candidate.id:
            restored = c
            break
    assert restored is not None, "candidate not found after replay"
    assert len(restored.edit_reports) == 1
    assert restored.edit_reports[0] == report


# ── guards and invariants ─────────────────────────────────────────────────

def test_text_patch_never_mutates_input():
    """input state is unchanged after apply."""
    state = _state("hello world")
    before = copy.deepcopy(state)
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    _apply(state, edit)
    assert state == before


def test_text_patch_frozen_knob_refused():
    """text_patch on a frozen knob raises EditError(code='E-knob')."""
    state = _state("hello", frozen=True)
    edit = _patch([{"kind": "append", "content": "more"}])
    with pytest.raises(EditError) as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-knob"


def test_text_patch_non_str_knob_refused():
    """text_patch on an int knob raises EditError(code='E-knob')."""
    state = _state("irrelevant", kind="int")
    edit = _patch([{"kind": "append", "content": "text"}])
    with pytest.raises(EditError) as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-knob"


def test_empty_target_skipped():
    """edit with empty target is skipped, not matched universally."""
    state = _state("hello world")
    edit = _patch([{"kind": "replace", "target": "", "content": "X"}])
    result, reports = _apply_with_report(state, edit)
    assert _value(result) == "hello world"
    assert reports[0]["outcomes"][0]["status"] == "skipped"


def test_empty_protected_region_marker_refused():
    """protected_regions with an empty-string marker raises EditError."""
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "hello", "content": "hi"}],
        config={"protected_regions": [["", "<!-- END -->"]]},
    )
    with pytest.raises(EditError) as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-patch"


# ── SKILLOPT_PATCH_CONFIG byte-exact tests (brief fix 2) ──────────────────
# Each test cites the SkillOpt source it reproduces.
# Source: microsoft/SkillOpt  skillopt/optimizer/skill.py

def test_skillopt_insert_after_formatting():
    """insert_after with SKILLOPT_PATCH_CONFIG on 'a\\nb' targeting 'a' with
    content 'X' produces 'a\\n\\nX\\nb' (blank-line separator).
    Cites: skillopt/optimizer/skill.py insert_after logic (~L80-92).
    """
    state = _state("a\nb")
    edit = _patch(
        [{"kind": "insert_after", "target": "a", "content": "X"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = _apply(state, edit)
    assert _value(result) == "a\n\nX\nb"


def test_skillopt_append_trailing_newline():
    """append with SKILLOPT_PATCH_CONFIG produces output ending in '\\n'.
    Cites: skillopt/optimizer/skill.py append semantics (~L70-78).
    """
    state = _state("hello")
    edit = _patch(
        [{"kind": "append", "content": "world"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = _apply(state, edit)
    val = _value(result)
    assert val.endswith("\n")
    assert val == "hello\n\nworld\n"


def test_skillopt_append_strips_trailing_whitespace():
    """Input 'hello   ' with append content 'X' under SKILLOPT_PATCH_CONFIG
    first strips trailing whitespace, then appends.
    Cites: skillopt/optimizer/skill.py append strips trailing ws (~L70).
    """
    state = _state("hello   ")
    edit = _patch(
        [{"kind": "append", "content": "X"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = _apply(state, edit)
    val = _value(result)
    # The trailing whitespace is stripped before appending
    assert val == "hello\n\nX\n"


def test_skillopt_content_whitespace_stripped():
    """Content '  earth  ' is stripped to 'earth' under SKILLOPT_PATCH_CONFIG.
    Cites: skillopt/optimizer/skill.py content is whitespace-stripped (~L65).
    """
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "world", "content": "  earth  "}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = _apply(state, edit)
    val = _value(result)
    # Content should be stripped: "  earth  " → "earth"
    assert val == "hello earth"
    assert "  earth  " not in val


def test_skillopt_append_before_protected_blank_line_separation():
    """Append before a protected region under SKILLOPT_PATCH_CONFIG produces
    blank-line separation ('\\n\\n') between content and the marker.
    Cites: skillopt/optimizer/skill.py protected region handling.
    """
    text = "content\n<!-- SLOW_UPDATE -->do not touch<!-- /SLOW_UPDATE -->"
    state = _state(text)
    edit = _patch(
        [{"kind": "append", "content": "NEW"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = _apply(state, edit)
    val = _value(result)
    prot_idx = val.find("<!-- SLOW_UPDATE -->")
    new_idx = val.find("NEW")
    assert new_idx < prot_idx
    # blank-line separation between content and protected region
    between = val[new_idx + len("NEW"):prot_idx]
    assert "\n\n" in between


def test_skillopt_insert_after_empty_target_falls_back_to_append():
    """insert_after with empty target under SKILLOPT_PATCH_CONFIG falls back
    to append.
    Cites: skillopt/optimizer/skill.py insert_after empty target fallback.
    """
    state = _state("hello")
    edit = _patch(
        [{"kind": "insert_after", "target": "", "content": "X"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result, reports = _apply_with_report(state, edit)
    val = _value(result)
    assert "X" in val
    assert reports[0]["outcomes"][0]["status"] == "applied_append"


def test_skillopt_unknown_edit_kind_skipped():
    """An unknown edit kind under SKILLOPT_PATCH_CONFIG is skipped (not refused).
    Cites: skillopt/optimizer/skill.py unknown edit kinds are silently skipped.
    """
    state = _state("hello")
    edit = _patch(
        [{"kind": "unknown_op", "target": "hello", "content": "X"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    # Should not raise
    result, reports = _apply_with_report(state, edit)
    assert _value(result) == "hello"  # unchanged
    assert reports[0]["outcomes"][0]["status"] == "skipped_unknown_kind"


def test_skillopt_status_names():
    """Status strings in the report match SkillOpt's names when
    SKILLOPT_PATCH_CONFIG is used.
    Cites: skillopt/optimizer/skill.py _apply_edit_with_report status names.
    """
    state = _state("hello world")
    edit = _patch(
        [
            {"kind": "replace", "target": "world", "content": "earth"},
            {"kind": "append", "content": "end"},
        ],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    _result, reports = _apply_with_report(state, edit)
    outcomes = reports[0]["outcomes"]
    assert outcomes[0]["status"] == "applied_replace"
    assert outcomes[1]["status"] == "applied_append"


def test_skillopt_default_protected_markers():
    """SKILLOPT_PATCH_CONFIG protects SLOW_UPDATE and APPENDIX markers by default.
    Cites: skillopt/optimizer/skill.py SLOW_UPDATE_HEADER / APPENDIX_HEADER.
    """
    text = "content\n<!-- SLOW_UPDATE -->secret<!-- /SLOW_UPDATE -->\nmore\n<!-- APPENDIX -->app<!-- /APPENDIX -->"
    state = _state(text)
    # Try to replace inside SLOW_UPDATE
    edit = _patch(
        [{"kind": "replace", "target": "secret", "content": "CHANGED"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result, reports = _apply_with_report(state, edit)
    assert _value(result) == text  # unchanged — protected
    assert reports[0]["outcomes"][0]["status"] == "skipped_protected"


# ── general defaults tests ────────────────────────────────────────────────

def test_general_defaults_refuse_unknown_kind():
    """The general default (no config) refuses an unknown edit kind with EditError."""
    state = _state("hello")
    edit = _patch([{"kind": "unknown_op", "target": "hello", "content": "X"}])
    with pytest.raises(EditError) as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-patch"


def test_general_insert_after_single_newline():
    """The general default insert_after uses '\\n' separator (not '\\n\\n')."""
    state = _state("a\nb")
    edit = _patch([{"kind": "insert_after", "target": "a", "content": "X"}])
    result = _apply(state, edit)
    assert _value(result) == "a\nX\nb"


# ── random optimizer exclusion (brief fix 3) ──────────────────────────────

def test_seeded_random_search_draws_same_ops_as_main():
    """With a fixed seed, random search draws exactly the same operations as
    main (text_patch is excluded from every preset, including 'broad').
    Fails if any preset includes text_patch.
    """
    import random as stdlib_random

    for preset in edits.PRESETS:
        ops = edits.enabled(preset)
        assert "text_patch" not in ops, f"text_patch found in preset {preset!r}"

    # Verify seeded random search produces deterministic results without text_patch
    rng1 = stdlib_random.Random(42)
    rng2 = stdlib_random.Random(42)
    state = _state("hello world")
    edit1 = edits.random_edit(state, preset="broad", rng=rng1)
    edit2 = edits.random_edit(state, preset="broad", rng=rng2)
    assert edit1 == edit2, "seeded random search is not deterministic"
