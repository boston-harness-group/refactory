"""Tests for set_knob's ``patch`` form, which applies text edits to a str knob (F7).

It was the ``text_patch`` primitive until it was folded into ``set_knob``.

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
    edit = {"op": "set_knob", "name": name, "patch": edits_list}
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

def test_patch_never_mutates_input():
    """input state is unchanged after apply."""
    state = _state("hello world")
    before = copy.deepcopy(state)
    edit = _patch([{"kind": "replace", "target": "world", "content": "earth"}])
    _apply(state, edit)
    assert state == before


def test_patch_frozen_knob_refused():
    """A patch on a frozen knob is refused exactly as a value is (E-knob)."""
    state = _state("hello", frozen=True)
    edit = _patch([{"kind": "append", "content": "more"}])
    with pytest.raises(EditError, match="frozen") as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-knob"
    with pytest.raises(EditError, match="frozen") as value_exc:
        _apply(state, {"op": "set_knob", "name": "prompt", "value": "more"})
    assert str(value_exc.value) == str(exc_info.value)


def test_patch_on_run_frozen_knob_refused():
    """A knob frozen for the run (``--frozen``) refuses a patch as it refuses a value."""
    from refactory_dsh.optim.base import Optimizer

    parent = Candidate.from_state(_state("hello"))
    for edit in (_patch([{"kind": "append", "content": "more"}]),
                 {"op": "set_knob", "name": "prompt", "value": "hello more"}):
        result = Optimizer().attempt(
            Proposal(edit=edit, parent=parent.id), parent, (), False, frozen=["prompt"],
        )
        assert result == ["E-frozen: knob 'prompt' is frozen for this run"]


def test_patch_non_str_knob_refused():
    """A patch on an int knob raises EditError(code='E-knob')."""
    state = _state("irrelevant", kind="int")
    edit = _patch([{"kind": "append", "content": "text"}])
    with pytest.raises(EditError, match="'patch' needs a str knob") as exc_info:
        _apply(state, edit)
    assert exc_info.value.code == "E-knob"


@pytest.mark.parametrize("fields", [
    {"value": "hello", "patch": [{"kind": "append", "content": "x"}]},
    {},
])
def test_set_knob_needs_exactly_one_of_value_and_patch(fields):
    """Both forms at once, or neither, is refused rather than guessed between."""
    edit = {"op": "set_knob", "name": "prompt", **fields}
    with pytest.raises(EditError, match="exactly one of 'value'") as exc_info:
        _apply(_state("hello"), edit)
    assert exc_info.value.code == "E-knob"


def test_set_knob_config_needs_a_patch():
    """``config`` configures a patch; with a value it is refused, not ignored."""
    edit = {"op": "set_knob", "name": "prompt", "value": "hi", "config": {"match": "exact"}}
    with pytest.raises(EditError, match="'config' configures a 'patch'"):
        _apply(_state("hello"), edit)


def test_set_knob_patch_must_be_a_list():
    with pytest.raises(EditError, match="'patch' must be a list") as exc_info:
        _apply(_state("hello"), {"op": "set_knob", "name": "prompt", "patch": "append x"})
    assert exc_info.value.code == "E-patch"


def test_text_patch_is_gone_and_says_where_it_went():
    """The old op is not in the catalogue; every refusal of it names set_knob's patch."""
    old = {"op": "text_patch", "name": "prompt",
           "edits": [{"kind": "append", "content": "x"}]}
    assert "text_patch" not in edits.names()
    assert all(op["op"] != "text_patch" for op in edits.catalogue())
    with pytest.raises(EditError, match="folded into set_knob") as exc_info:
        _apply(_state("hello"), old)
    assert exc_info.value.code == "E-op"
    assert "folded into set_knob" in edits.precheck(old, _state("hello"))
    assert edits.unknown("text_patch").startswith("E-op: unknown edit 'text_patch': ")
    assert "'patch'" in edits.unknown("text_patch")
    # Any other unknown op keeps the plain message.
    assert edits.unknown("nonsense") == f"E-op: unknown edit 'nonsense'; known: {edits.names()}"
    from refactory_dsh.optim.base import Optimizer
    parent = Candidate.from_state(_state("hello"))
    result = Optimizer().attempt(Proposal(edit=old, parent=parent.id), parent, (), False)
    assert isinstance(result, list) and "folded into set_knob" in result[0]


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


# ── SKILLOPT_PATCH_CONFIG byte-exact tests ────────────────────────────────
# Every expected value below is a literal copied from running microsoft/SkillOpt's
# own code: skillopt/optimizer/skill.py apply_patch_with_report() (MIT License,
# Copyright (c) Microsoft Corporation) on the same input, with "op" for "kind".
# Line numbers cite that file.

_SU = "<!-- SLOW_UPDATE_START -->s<!-- SLOW_UPDATE_END -->"  # skill.py:14-15


def _skillopt(text: str, edits_list: list[dict]) -> tuple[str, list[str]]:
    """Apply under SKILLOPT_PATCH_CONFIG; return (text, statuses)."""
    result, reports = _apply_with_report(
        _state(text), _patch(edits_list, config=dict(SKILLOPT_PATCH_CONFIG)),
    )
    return _value(result), [o["status"] for o in reports[0]["outcomes"]]


@pytest.mark.parametrize(
    ("text", "edits_list", "expected_text", "expected_statuses"),
    [
        # skill.py:118-122 insert_after: content on its own line after the target's line.
        ("a\nb", [{"kind": "insert_after", "target": "a", "content": "X"}],
         "a\n\nX\nb", ["applied_insert_after"]),
        # skill.py:120 last line without a newline: insert_at = len(skill).
        ("a target", [{"kind": "insert_after", "target": "target", "content": "X"}],
         "a target\nX\n", ["applied_insert_after"]),
        # skill.py:105-106 append: rstrip, blank line, content, trailing newline.
        ("hello   ", [{"kind": "append", "content": "X"}],
         "hello\n\nX\n", ["applied_append"]),
        # skill.py:79 content is .strip()ped.
        ("abc", [{"kind": "replace", "target": "b", "content": "  X  \n"}],
         "aXc", ["applied_replace"]),
        # skill.py:78-79 strip first, then markers removed: the newline survives.
        ("abc", [{"kind": "replace", "target": "b",
                  "content": "X\n<!-- SLOW_UPDATE_START -->"}],
         "aX\nc", ["applied_replace"]),
        # skill.py:99-104 append before the protected region; its tail is kept as is.
        ("body  \n" + _SU + "  \n", [{"kind": "append", "content": "X"}],
         "body\n\nX\n\n" + _SU + "  \n", ["applied_append_before_protected_region"]),
        # skill.py:109-115 missing insert_after target with a protected region.
        ("body\n" + _SU, [{"kind": "insert_after", "target": "zz", "content": "X"}],
         "body\n\nX\n\n" + _SU, ["applied_insert_after_fallback_before_protected_region"]),
        # skill.py:109, 116-117 empty insert_after target, no protected region.
        ("abc  ", [{"kind": "insert_after", "target": "", "content": "X"}],
         "abc\n\nX\n", ["applied_insert_after_fallback_append"]),
        # skill.py:94-96 target inside a protected region.
        ("body\n" + _SU, [{"kind": "replace", "target": "s", "content": "X"}],
         "body\n" + _SU, ["skipped_protected_region"]),
        # skill.py:94-96 the protected check precedes dispatch, so append is skipped too.
        ("body\n" + _SU, [{"kind": "append", "target": "s", "content": "X"}],
         "body\n" + _SU, ["skipped_protected_region"]),
        # skill.py:56 only the target's start is tested: a target running into the
        # region is applied.
        ("body\n" + _SU, [{"kind": "replace", "target": "body\n<!-- SLOW", "content": "X"}],
         "X_UPDATE_START -->s<!-- SLOW_UPDATE_END -->", ["applied_replace"]),
        # skill.py:125-145 skip statuses for each op.
        ("abc", [
            {"kind": "bogus", "target": "a", "content": "X"},
            {"kind": "replace", "target": "", "content": "X"},
            {"kind": "replace", "target": "z", "content": "X"},
            {"kind": "delete", "target": ""},
            {"kind": "delete", "target": "z"},
            {"kind": "delete", "target": "b"},
        ], "ac", [
            "skipped_unknown_op", "skipped_replace_missing_target",
            "skipped_replace_target_not_found", "skipped_delete_missing_target",
            "skipped_delete_target_not_found", "applied_delete",
        ]),
        # skill.py:173-184 a failing edit is reported as "error" and the patch goes on.
        ("a\nb", [
            {"kind": "replace", "target": "a", "content": 5},
            {"kind": "insert_after", "target": "b", "content": " Y "},
        ], "a\nb\nY\n", ["error", "applied_insert_after"]),
    ],
)
def test_skillopt_config_matches_skillopt(text, edits_list, expected_text, expected_statuses):
    assert _skillopt(text, edits_list) == (expected_text, expected_statuses)


def test_skillopt_default_protected_markers():
    """SKILLOPT_PATCH_CONFIG protects exactly SkillOpt's marker pairs (skill.py:14-20)."""
    assert SKILLOPT_PATCH_CONFIG["protected_regions"] == [
        ["<!-- SLOW_UPDATE_START -->", "<!-- SLOW_UPDATE_END -->"],
        ["<!-- APPENDIX_START -->", "<!-- APPENDIX_END -->"],
    ]


def test_attempt_stores_patch_report_on_candidate():
    """Optimizer.attempt keeps the patch report on candidate.edit_reports."""
    from refactory_dsh.optim.base import Optimizer

    parent = Candidate.from_state(_state("a\nb"))
    edit = _patch(
        [{"kind": "insert_after", "target": "a", "content": "X"}],
        config=dict(SKILLOPT_PATCH_CONFIG),
    )
    result = Optimizer().attempt(Proposal(edit=edit, parent=parent.id), parent, (), False)
    assert isinstance(result, Candidate), result
    assert result.state["knobs"]["prompt"]["value"] == "a\n\nX\nb"
    assert result.edits == [edit]
    assert len(result.edit_reports) == 1
    report = result.edit_reports[0]
    assert report is not None
    assert [o["status"] for o in report["outcomes"]] == ["applied_insert_after"]
    assert report["noop"] is False


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


# ── random search never samples the patch form (brief fix 3) ─────────────

#: ``broad`` as it was before the patch form existed: the catalogue random search
#: draws its op from, so a seeded run draws the same op sequence.
_BROAD_BEFORE_PATCH = [
    "add_edge", "add_knob", "add_node", "bypass", "duplicate", "insert_after",
    "insert_before", "loop", "remove_edge", "remove_knob", "remove_node", "reorder",
    "set_knob", "split", "wrap",
]


def test_seeded_random_search_draws_same_ops_as_main():
    """Seeded random search is unchanged by the patch form: it draws from the same
    op list as before, and the only knob edit it builds is a ``value``.
    """
    import random as stdlib_random

    # `parallel` came later and is never drawn (`edits._NOT_SAMPLED`), so the list the
    # draw chooses from is still the one before the patch form.
    for preset in ("broad", "structure"):
        drawable = [op for op in edits.enabled(preset) if op not in edits._NOT_SAMPLED]
        assert drawable == _BROAD_BEFORE_PATCH

    state = _state("hello world")
    state["knobs"]["n"] = {"value": 3, "spec": {"kind": "int", "default": 3, "lo": 0, "hi": 9}}
    drawn = []
    for seed in range(200):
        edit = edits.random_edit(state, preset="broad", rng=stdlib_random.Random(seed))
        if edit is not None:
            drawn.append(edit)
    knob_edits = [e for e in drawn if e["op"] == "set_knob"]
    assert {e["name"] for e in knob_edits} == {"prompt", "n"}
    assert all(set(e) == {"op", "name", "value"} for e in knob_edits), knob_edits

    # Deterministic for a fixed seed.
    assert edits.random_edit(state, rng=stdlib_random.Random(42)) == edits.random_edit(
        state, rng=stdlib_random.Random(42))
