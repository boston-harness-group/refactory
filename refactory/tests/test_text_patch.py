"""Tests for the text_patch edit primitive (F7).

Each test fails when the feature it tests is reverted.
"""

from __future__ import annotations

import copy
import json

import pytest

from refactory_dsh.nn import edits
from refactory_dsh.nn.edits import EditError, SKILLOPT_PATCH_CONFIG, apply


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
    result = _apply(state, edit)
    assert _value(result) == "hello world"
    assert edit["_report"]["outcomes"][0]["status"] == "skipped_replace_target_not_found"


def test_on_missing_append():
    """target not found with on_missing='append' appends content."""
    state = _state("hello world")
    edit = _patch(
        [{"kind": "replace", "target": "MISSING", "content": "APPENDED"}],
        config={"on_missing": "append"},
    )
    result = _apply(state, edit)
    assert "APPENDED" in _value(result)
    assert edit["_report"]["outcomes"][0]["status"] == "applied_append"


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
    result = _apply(state, edit)
    val = _value(result)
    assert "FALLBACK" in val
    assert "NOPE" not in val
    outcomes = edit["_report"]["outcomes"]
    assert outcomes[0]["status"] == "applied_append"
    assert outcomes[1]["status"] == "skipped_replace_target_not_found"


def test_skillopt_patch_config():
    """SKILLOPT_PATCH_CONFIG: missing insert_after → append, missing replace → skip."""
    state = _state("hello world")
    edit = _patch(
        [
            {"kind": "insert_after", "target": "NONEXISTENT", "content": "FALLBACK_IA"},
            {"kind": "replace", "target": "NONEXISTENT", "content": "SHOULD_SKIP"},
            {"kind": "delete", "target": "NONEXISTENT"},
        ],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = _apply(state, edit)
    val = _value(result)
    assert "FALLBACK_IA" in val
    assert "SHOULD_SKIP" not in val
    outcomes = edit["_report"]["outcomes"]
    assert outcomes[0]["status"] == "applied_append"
    assert outcomes[1]["status"] == "skipped_replace_target_not_found"
    assert outcomes[2]["status"] == "skipped_delete_target_not_found"


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
    result = _apply(state, edit)
    assert _value(result) == text  # unchanged
    assert edit["_report"]["outcomes"][0]["status"] == "skipped_protected_region"


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
    """multi-edit patch returns report list with index, kind, status per edit."""
    state = _state("alpha beta gamma")
    edit = _patch([
        {"kind": "replace", "target": "beta", "content": "BETA"},
        {"kind": "delete", "target": "MISSING"},
        {"kind": "append", "content": "tail"},
    ])
    _apply(state, edit)
    report = edit["_report"]
    assert len(report["outcomes"]) == 3
    assert report["outcomes"][0]["index"] == 0
    assert report["outcomes"][0]["kind"] == "replace"
    assert report["outcomes"][0]["status"] == "applied_replace"
    assert report["outcomes"][1]["index"] == 1
    assert report["outcomes"][1]["status"] == "skipped_replace_target_not_found" or \
           report["outcomes"][1]["status"] == "skipped_delete_target_not_found"
    assert report["outcomes"][2]["index"] == 2
    assert report["outcomes"][2]["status"] == "applied_append"


def test_outcome_report_persisted_in_edit_record():
    """The _report dict on the edit is serialisable (for journaling)."""
    state = _state("hello world")
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    _apply(state, edit)
    report = edit["_report"]
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
    _apply(state, edit)
    assert edit["_report"]["noop"] is True


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
    result = _apply(state, edit)
    assert _value(result) == "hello world"
    assert edit["_report"]["outcomes"][0]["status"] == "skipped"
