"""SkillOpt over several text knobs: the composite document and its mapping back.

Coverage:
- one knob (the default) is seen, routed and patched exactly as before
- the composite: a header naming the graph's agent steps, then each knob's text
  verbatim under its SECTION line, the last knob ending the document
- routing: an edit goes to the one knob holding its target; an append or an
  unfound target to the last knob; a target in several knobs is refused unless the
  edit names its section, and a named section takes the edit alone (refused when
  its target is elsewhere, an append or unfound target landing there); a target across a section boundary, in the
  header or in a SECTION line, and content carrying a SECTION line, are refused
  per edit while the rest of the patch applies; later edits route against what
  earlier ones wrote
- knobs that are missing, frozen or not text are refused before anything is spent
- livemath/skill_graph with its four system-prompt knobs and a scripted model (no paid
  calls): analysts -> merge -> patch -> val gate -> slow update -> meta skill, the
  edits landing on the right knobs and reaching the step that reads each; unique,
  replayable call keys; resume from a mid-epoch kill
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import strategies
from refactory_dsh.loop import LoopConfig
from refactory_dsh.optim import Candidate
from refactory_dsh.outerloop import ValidationGate

from refactory_packages.skillopt import prompts
from refactory_packages.skillopt.loop import SkillOptLoop
from refactory_packages.skillopt.proposer import (
    AMBIGUOUS_SECTION, CROSS_SECTION, SECTION_LINE_IN_CONTENT, TARGET_NOT_IN_SECTION,
    UNKNOWN_SECTION, SkillOptProposer, composite,
)
from refactory_packages.skillopt.trace import current_call_key

from test_livemath_blocks import _patch_task
from test_skillopt import FakeClient, ProcessKilled

MODE = "livemath/skill_graph"
KNOBS = ("compare__system_prompt", "precision__system_prompt", "hypotheses__system_prompt", "decide__system_prompt")
TRAIN = ["202501:1", "202501:2"]   # 202501:1 is B (passes), 202501:2 is not (fails)
VAL = ["202503:1"]


def _seed_state() -> dict[str, Any]:
    from refactory_packages.livemath.skill_graph import SkillGraph

    return SkillGraph().state_dict()


def _candidate(state: dict[str, Any]) -> Candidate:
    return Candidate.from_state(state, parent=None, edits=[])


def _proposer(knobs: tuple[str, ...] = KNOBS) -> SkillOptProposer:
    return SkillOptProposer(knobs=knobs)


def _apply(per_knob, ranked, state):  # type: ignore[no-untyped-def]
    """Each knob's routed edits applied as that knob's own set_knob patch."""
    from refactory_dsh.nn import edits as nn_edits

    proposer = _proposer()
    for name, indices in per_knob:
        edit = proposer.patch([ranked[i] for i in indices], _candidate(state), "", knob=name).edit
        state = nn_edits.apply(edit, state)
    return state


# ── one knob: exactly as before ───────────────────────────────────────────


def test_one_knob_is_seen_and_patched_exactly_as_before():
    """With the default single knob the text is the knob's value, every edit goes to
    it (a SECTION-looking target or content included: there are no sections), and
    the patch is one set_knob of all the ranked edits. Fails when the composite or
    the routing is applied to a single knob."""
    state = {"graph": {"nodes": {}}, "knobs": {"skill": {
        "value": "## A\nalpha <!-- SECTION x: y -->\n", "spec": {"kind": "str"}}}}
    proposer = SkillOptProposer()
    parent = _candidate(state)
    assert proposer.knobs == ("skill",) and not proposer.multi
    assert proposer.text(parent) == "## A\nalpha <!-- SECTION x: y -->\n"
    ranked = [
        {"op": "replace", "target": "alpha <!-- SECTION x: y -->", "content": "beta"},
        {"op": "append", "content": "<!-- SECTION z: w -->"},
    ]
    assert proposer.route(ranked, parent) == ([("skill", [0, 1])], [])
    edit = proposer.patch(ranked, parent, "r").edit
    assert edit["name"] == "skill" and [e["kind"] for e in edit["patch"]] == ["replace", "append"]
    assert proposer.state_dict()["proposer_knobs"] == ["skill"]


# ── the composite ─────────────────────────────────────────────────────────


def test_composite_lists_each_knob_under_its_section_line():
    from refactory_packages.livemath.skill_graph import step_system_prompt

    state = _seed_state()
    text = _proposer().text(_candidate(state))
    assert text.startswith("# Skill sections\n")
    assert ("agent steps, in order (| marks steps that run in parallel): "
            "compare -> hypotheses | precision -> decide.") in text
    positions = []
    for knob, step in zip(KNOBS, ("compare", "precision", "hypotheses", "decide")):
        line = f"<!-- SECTION {knob}: system prompt for the {step} step -->"
        assert f"{line}\n{step_system_prompt(step)}" in text
        positions.append(text.index(line))
    assert positions == sorted(positions)
    assert text.endswith(step_system_prompt("decide"))


# ── routing ───────────────────────────────────────────────────────────────


def test_edits_route_to_the_section_holding_their_target():
    """Each edit lands in the knob whose text holds its target; an append and an
    unfound target go to the last knob. Fails when routing ignores the target."""
    state = _seed_state()
    ranked = [
        {"op": "replace", "target": "Compare all options before committing.",
         "content": "CMP_EDIT compare every option."},
        {"op": "insert_after", "target": "## Hypotheses", "content": "- HYP_EDIT check domains."},
        {"op": "append", "content": "- APPEND_EDIT answer with the label."},
        {"op": "replace", "target": "text that is nowhere", "content": "LOST"},
        {"op": "insert_after", "target": "also nowhere", "content": "- FALLBACK_EDIT"},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert refused == []
    assert per_knob == [("compare__system_prompt", [0]), ("hypotheses__system_prompt", [1]),
                        ("decide__system_prompt", [2, 3, 4])]
    out = _apply(per_knob, ranked, state)["knobs"]
    assert "CMP_EDIT" in out["compare__system_prompt"]["value"]
    assert "## Hypotheses\n\n- HYP_EDIT" in out["hypotheses__system_prompt"]["value"]
    decide = out["decide__system_prompt"]["value"]
    assert "APPEND_EDIT" in decide and "FALLBACK_EDIT" in decide and "LOST" not in decide
    assert out["precision__system_prompt"] == state["knobs"]["precision__system_prompt"]


def test_cross_section_targets_are_refused_and_the_rest_applies():
    """A target across a section boundary, in the header or in a SECTION line, and
    content carrying a SECTION line, are refused one by one; the other edits still
    route. Fails when a cross-section edit is routed (or splits) instead."""
    from refactory_packages.livemath.skill_graph import step_system_prompt

    state = _seed_state()
    compare_tail = step_system_prompt("compare")[-30:]
    ranked = [
        # The end of compare's text, the blank line and precision's SECTION line.
        {"op": "replace", "target": f"{compare_tail}\n\n<!-- SECTION precision__system_prompt",
         "content": "X"},
        {"op": "delete", "target": "<!-- SECTION hypotheses__system_prompt: system prompt"},
        {"op": "replace", "target": "# Skill sections", "content": "# HEADER_EDIT"},
        {"op": "append", "content": "<!-- SECTION forged: knob -->\n- FORGED"},
        {"op": "replace", "target": "## Final Answer", "content": "## Final Answer (OK_EDIT)"},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert [(r["index"], r["status"], r["knob"]) for r in refused] == [
        (0, CROSS_SECTION, None), (1, CROSS_SECTION, None), (2, CROSS_SECTION, None),
        (3, SECTION_LINE_IN_CONTENT, None),
    ]
    assert per_knob == [("decide__system_prompt", [4])]
    out = _apply(per_knob, ranked, state)["knobs"]
    assert "OK_EDIT" in out["decide__system_prompt"]["value"]
    for knob in KNOBS[:3]:
        assert out[knob] == state["knobs"][knob]


def test_an_unnamed_target_in_several_sections_is_refused_naming_them():
    """"## Task Format" is in every step's system prompt: an edit that does not name
    its section is refused, its outcome naming all four candidate sections, and the
    rest of the patch applies. Fails on a silent first match."""
    state = _seed_state()
    for knob in KNOBS:
        assert "## Task Format" in state["knobs"][knob]["value"], knob
    ranked = [
        {"op": "replace", "target": "## Task Format", "content": "## Input Format"},
        {"op": "replace", "target": "## Final Answer", "content": "## Final Answer (OK_EDIT)"},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert per_knob == [("decide__system_prompt", [1])]
    assert [(r["index"], r["status"], r["sections"]) for r in refused] == [
        (0, AMBIGUOUS_SECTION, list(KNOBS))]
    assert all(knob in refused[0]["message"] for knob in KNOBS)
    assert '"section"' in refused[0]["message"]
    run = _AttemptRun()
    outcome, report = _proposer().attempt_patch(run, ranked, _candidate(state), "r")
    assert report["outcomes"][0]["status"] == AMBIGUOUS_SECTION
    knobs = outcome.state["knobs"]
    assert "OK_EDIT" in knobs["decide__system_prompt"]["value"]
    assert all("## Input Format" not in knobs[k]["value"] for k in KNOBS)


def test_a_named_section_takes_a_shared_target_alone():
    """The same shared target named to decide's section changes decide's copy only,
    and the section field never reaches the knob's patch. Fails when a named edit
    goes to the first knob holding the target."""
    state = _seed_state()
    ranked = [{"op": "replace", "target": "## Task Format", "content": "## Input Format",
               "section": "decide__system_prompt"}]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert refused == [] and per_knob == [("decide__system_prompt", [0])]
    patch = _proposer().patch(ranked, _candidate(state), "", knob="decide__system_prompt").edit
    assert all("section" not in item for item in patch["patch"])
    out = _apply(per_knob, ranked, state)["knobs"]
    assert "## Input Format" in out["decide__system_prompt"]["value"]
    for knob in KNOBS[:3]:
        assert out[knob] == state["knobs"][knob]


def test_a_named_section_refuses_a_target_found_only_elsewhere():
    """A named edit whose target is in another section (or only in the structure) is
    refused, naming where the target is. Fails when it falls back to that section."""
    state = _seed_state()
    ranked = [
        {"op": "replace", "target": "Compare all options before committing.",
         "content": "X", "section": "decide__system_prompt"},
        {"op": "insert_after", "target": "# Skill sections", "content": "- Y",
         "section": "precision__system_prompt"},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert per_knob == []
    assert [(r["index"], r["status"], r["sections"]) for r in refused] == [
        (0, TARGET_NOT_IN_SECTION, ["compare__system_prompt"]),
        (1, TARGET_NOT_IN_SECTION, []),
    ]
    assert "decide__system_prompt" in refused[0]["message"]
    assert "compare__system_prompt" in refused[0]["message"]
    # A target in no section's text says where it is, not "it occurs in: none".
    assert "across a section boundary" in refused[1]["message"]
    assert "none" not in refused[1]["message"]


def test_a_named_append_or_unfound_target_lands_at_the_end_of_that_section():
    """An append and an insert_after whose target is nowhere, both naming precision's
    section, end precision's text (SkillOpt's missing-target rule there), not the
    last knob's. Fails when a named append still goes to the last knob."""
    state = _seed_state()
    ranked = [
        {"op": "append", "content": "- NAMED_APPEND", "section": "precision__system_prompt"},
        {"op": "insert_after", "target": "nowhere at all", "content": "- NAMED_FALLBACK",
         "section": "precision__system_prompt"},
        {"op": "append", "content": "- UNNAMED_APPEND"},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert refused == []
    assert per_knob == [("precision__system_prompt", [0, 1]), ("decide__system_prompt", [2])]
    out = _apply(per_knob, ranked, state)["knobs"]
    assert out["precision__system_prompt"]["value"].rstrip().endswith(
        "- NAMED_APPEND\n\n- NAMED_FALLBACK")
    assert out["decide__system_prompt"]["value"].rstrip().endswith("- UNNAMED_APPEND")


@pytest.mark.parametrize("section", ["nope__system_prompt", "data_parallelism", 3])
def test_a_section_naming_no_knob_is_refused(section):
    state = _seed_state()
    ranked = [{"op": "append", "content": "- Z", "section": section}]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert per_knob == []
    assert [(r["status"], r["sections"]) for r in refused] == [(UNKNOWN_SECTION, list(KNOBS))]


def test_one_knob_prompts_never_carry_the_section_field():
    """The section note and the ranker's section column are multi-knob only: with
    one knob a system prompt is unchanged and an edit's section is not shown."""
    edit = {"op": "append", "content": "x", "support_count": 1, "section": "s"}
    plain = {k: v for k, v in edit.items() if k != "section"}
    assert prompts.describe_edit(edit) == prompts.describe_edit(plain)
    assert "section='s'" in prompts.describe_edit(edit, sections=True)
    assert prompts.with_sections(prompts.RANK_SYSTEM, False) == prompts.RANK_SYSTEM
    assert prompts.with_sections(prompts.RANK_SYSTEM, True).endswith(prompts.SECTION_EDIT_NOTE)
    assert prompts.format_rank_user("S", [edit], 1) == prompts.format_rank_user("S", [plain], 1)


def test_the_header_asks_for_a_section_on_shared_targets():
    from refactory_packages.skillopt.proposer import composite_header

    state = _seed_state()
    header = composite_header(state)
    assert _proposer().text(_candidate(state)).startswith(header + "\n\n")
    assert '"section": "<knob>"' in header
    assert "appears in more than one section must name its section" in header


def test_an_insert_after_aimed_at_the_structure_appends_to_the_last_knob():
    """An insert_after whose target is only in the header or a SECTION line follows
    SkillOpt's missing-target rule (append), in the last knob; one whose target
    crosses a section boundary is still refused. Fails when the structure-only
    insert_after is refused or lands elsewhere."""
    state = _seed_state()
    ranked = [
        {"op": "insert_after", "target": "# Skill sections", "content": "- HEADER_RULE"},
        {"op": "insert_after", "target": "<!-- SECTION precision__system_prompt", "content": "- LINE_RULE"},
        {"op": "insert_after", "target": "step decides.\n\n\n<!-- SECTION", "content": "- X"},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert per_knob == [("decide__system_prompt", [0, 1])]
    assert [(r["index"], r["status"]) for r in refused] == [(2, CROSS_SECTION)]
    run = _AttemptRun()
    outcome, report = _proposer().attempt_patch(run, ranked, _candidate(state), "r")
    decide = outcome.state["knobs"]["decide__system_prompt"]["value"]
    assert decide.endswith("- HEADER_RULE\n\n- LINE_RULE\n")
    assert [o["status"] for o in report["outcomes"]][:2] == [
        "applied_insert_after_fallback_append"] * 2
    for knob in KNOBS[:3]:
        assert outcome.state["knobs"][knob] == state["knobs"][knob]


def test_later_edits_route_against_what_earlier_edits_wrote():
    """An edit whose target an earlier edit of the same patch wrote goes to that
    knob. Fails when every edit is routed against the parent's texts."""
    state = _seed_state()
    ranked = [
        {"op": "insert_after", "target": "## Theorem-Level Precision", "content": "- NEW_RULE one."},
        {"op": "replace", "target": "- NEW_RULE one.", "content": "- NEW_RULE two."},
    ]
    per_knob, refused = _proposer().route(ranked, _candidate(state))
    assert refused == [] and per_knob == [("precision__system_prompt", [0, 1])]
    assert "NEW_RULE two." in _apply(per_knob, ranked, state)["knobs"]["precision__system_prompt"]["value"]


class _AttemptRun:
    """``Run.attempt``'s contract over plain states: apply the edit, refuse a state
    identical to its parent (``E-noop``) or one already held (``E-duplicate``)."""

    def __init__(self, held: tuple[Candidate, ...] = ()) -> None:
        self.held = {c.id: c for c in held}
        self.attempts: list[tuple[str, str]] = []   # (knob, parent id)
        #: What Run.attempt adds to the loop's rejections: every refusal.
        self.rejections: list[list[str]] = []

    def attempt(self, proposal, parent):  # type: ignore[no-untyped-def]
        from refactory_dsh.nn import edits as nn_edits
        from refactory_dsh.optim import candidate_id

        self.attempts.append((proposal.edit["name"], parent.id))
        state = nn_edits.apply(proposal.edit, parent.state)
        if candidate_id(state) == parent.id:
            self.rejections.append(["E-noop: unchanged"])
            return ["E-noop: unchanged"]
        if candidate_id(state) in self.held:
            self.rejections.append(["E-duplicate: held"])
            return ["E-duplicate: held"]
        return Candidate.from_state(state, parent=parent.id, edits=[*parent.edits, proposal.edit])

    def candidate(self, candidate_id):  # type: ignore[no-untyped-def]
        return self.held.get(candidate_id)


def _with_guidance(state: dict[str, Any]) -> dict[str, Any]:
    from refactory_dsh.nn import edits as nn_edits

    proposer = _proposer()
    proposal, _ = proposer.slow_update_proposal("GUIDE ME.", _candidate(state))
    return nn_edits.apply(proposal.edit, state)


def test_a_knob_whose_edits_change_nothing_is_passed_over():
    """compare's edit applies; decide's only edit targets its protected region and
    is skipped, so decide's patch changes nothing and is passed over without an
    attempt (no refusal recorded): the step still yields the compare change, as a
    child of the parent, and the report shows both. Fails when an E-noop knob
    refuses the whole step or adds a refusal."""
    state = _with_guidance(_seed_state())
    parent = _candidate(state)
    ranked = [{"op": "replace", "target": "## Option Comparison", "content": "## Options (CMP)"},
              {"op": "replace", "target": "GUIDE ME.", "content": "HIJACK"}]
    run = _AttemptRun()
    outcome, report = _proposer().attempt_patch(run, ranked, parent, "r")
    assert isinstance(outcome, Candidate)
    assert [knob for knob, _ in run.attempts] == ["compare__system_prompt"]
    assert run.rejections == []
    assert outcome.parent == parent.id and len(outcome.edits) == 1
    assert "(CMP)" in outcome.state["knobs"]["compare__system_prompt"]["value"]
    assert outcome.state["knobs"]["decide__system_prompt"] == state["knobs"]["decide__system_prompt"]
    assert [(o["knob"], o["status"]) for o in report["outcomes"]] == [
        ("compare__system_prompt", "applied_replace"), ("decide__system_prompt", "skipped_protected_region")]


def test_a_step_whose_every_edit_is_refused_is_a_noop():
    parent = _candidate(_seed_state())
    ranked = [{"op": "delete", "target": "<!-- SECTION decide__system_prompt"}]
    outcome, report = _proposer().attempt_patch(_AttemptRun(), ranked, parent, "r")
    assert isinstance(outcome, list) and outcome[0].startswith("E-noop")
    assert report["noop"] and report["outcomes"][0]["status"] == CROSS_SECTION


def test_a_step_whose_knob_patches_all_change_nothing_records_one_refusal():
    """Two knobs' patches that change nothing: the step is E-noop with exactly one
    refusal recorded, as one knob's patch records one. Fails when each knob adds
    its own refusal."""
    parent = _candidate(_seed_state())
    ranked = [{"op": "replace", "target": "## Option Comparison", "content": "## Option Comparison"},
              {"op": "replace", "target": "## Final Answer", "content": "## Final Answer"}]
    run = _AttemptRun()
    outcome, report = _proposer().attempt_patch(run, ranked, parent, "r")
    assert outcome == ["E-noop: unchanged"] and report["noop"]
    assert run.rejections == [["E-noop: unchanged"]]


def test_a_knob_patch_reaching_a_held_state_continues_from_it():
    """When compare's patch reaches a candidate the search holds, decide's patch is
    attempted on that candidate; when it is the step's last patch, the step is a
    duplicate, refused as with one knob. Fails when the held candidate is not
    looked up, or a duplicate result is accepted."""
    from refactory_dsh.nn import edits as nn_edits

    parent = _candidate(_seed_state())
    compare_edit = {"op": "replace", "target": "## Option Comparison", "content": "## Options (CMP)"}
    held_state = nn_edits.apply(_proposer().patch([compare_edit], parent, "", knob="compare__system_prompt").edit,
                                parent.state)
    held = Candidate.from_state(held_state, parent=parent.id, edits=[])
    run = _AttemptRun((held,))
    outcome, _ = _proposer().attempt_patch(
        run, [compare_edit, {"op": "append", "content": "- END"}], parent, "r")
    assert run.attempts == [("compare__system_prompt", parent.id), ("decide__system_prompt", held.id)]
    # The step's candidate is rebuilt as the parent's child, carrying both edits.
    assert isinstance(outcome, Candidate) and outcome.parent == parent.id
    assert [e["name"] for e in outcome.edits] == ["compare__system_prompt", "decide__system_prompt"]
    refused, _ = _proposer().attempt_patch(_AttemptRun((held,)), [compare_edit], parent, "r")
    assert refused == ["E-duplicate: held"]


def test_the_slow_update_lives_at_the_end_of_the_last_knob():
    """The placeholder and the guidance are a value of the last knob alone, so the
    composite ends with SkillOpt's protected region. Fails when they go to another
    knob or rewrite the whole composite."""
    from refactory_dsh.nn import edits as nn_edits

    state = _seed_state()
    proposer = _proposer()
    placeholder = proposer.placeholder(_candidate(state)).edit
    assert placeholder["name"] == "decide__system_prompt" and "value" in placeholder
    state = nn_edits.apply(placeholder, state)
    proposal, text = proposer.slow_update_proposal("GUIDE well.", _candidate(state))
    assert proposal.edit["name"] == "decide__system_prompt"
    after = nn_edits.apply(proposal.edit, state)
    assert after["knobs"]["decide__system_prompt"]["value"] == prompts.replace_slow_update_field(
        state["knobs"]["decide__system_prompt"]["value"], "GUIDE well.")
    assert text == composite(after, KNOBS)
    assert text.endswith(f"{prompts.SLOW_UPDATE_START}\nGUIDE well.\n{prompts.SLOW_UPDATE_END}\n")
    assert prompts.extract_slow_update_field(text) == "GUIDE well."
    for knob in KNOBS[:3]:
        assert after["knobs"][knob] == state["knobs"][knob]


def test_livemath_view_labels_each_step_and_shows_the_first_steps_system_prompt():
    """A skill_graph record's messages carry the step that wrote them, and the target
    system prompt is compare's system-prompt knob; a one-node (solver) record is
    unlabelled. Fails when the view drops the node or reads the absent ``skill`` knob."""
    from refactory_packages.livemath.skill_graph import STEPS
    from refactory_packages.livemath.skillopt_view import build_system, rollout_result
    from refactory_packages.skillopt.prompts import fmt_trajectory

    meta = {"question": "Q?", "choices": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"}],
            "correct_choice": {"label": "B", "text": "b"}}
    knobs = {f"{s}__system_prompt": f"SYSTEM-{s}" for s in STEPS}
    record = {"id": "x", "passed": True, "metadata": meta, "exchanges": [
        {"node": "compare", "prompt": "P", "response": "cmp"},
        {"node": "precision", "prompt": "P2", "response": "prc"},
        {"node": "decide", "prompt": "P3", "response": "<answer>B</answer>"},
    ]}
    result = rollout_result(record, knobs)
    assert result["target_system_prompt"] == "SYSTEM-compare"
    assert fmt_trajectory(result["conversation"]).startswith(
        "[compare] cmp\n[precision] prc\n[decide] <answer>B</answer>\n[verification]")
    solver = {**record, "exchanges": [{"node": "solve", "prompt": "P", "response": "<answer>B</answer>"}]}
    one = rollout_result(solver, {"skill": "S"})
    assert one["target_system_prompt"] == build_system("S")
    assert "role" not in one["conversation"][0]


# ── one knob (arm B): the optimizer's prompts are unchanged ───────────────

#: sha256 of every optimizer call (call key, system, prompt) of a two-epoch scripted
#: single-knob run (``test_skillopt._run_loop``), recorded on main before the
#: section field existed. Fails when anything multi-knob leaks into arm B's prompts.
ARM_B_PROMPTS_SHA256 = "351dfe56211ecf59bcb0dc8f7627fc8c3274eff161b1a55b3f954e5ac9f634d4"


def test_single_knob_optimizer_prompts_are_byte_identical(tmp_path, monkeypatch):
    """The messages the strategy builds, before the Claude backend framing every call
    gets (its own tests: test_skillopt, test_replay)."""
    import hashlib

    from refactory_packages.skillopt.loop import SkillOptLoop
    from test_skillopt import _run_loop

    monkeypatch.setattr(SkillOptLoop, "_optimizer_client", lambda self: self.model)
    client = FakeClient()
    _run_loop(tmp_path, client=client, config_overrides={"batch_size": 8},
              skillopt_options={"max_epochs": 2, "group_size": 2})
    calls = [[list(c["call_key"] or ()), c["system"], c["prompt"]] for c in client.calls]
    assert {k[0] for k, _, _ in calls} >= {
        "analyst_failure", "merge_failure", "slow_update", "meta_skill"}, calls
    assert not any("SECTION" in p for _, _, p in calls)
    digest = hashlib.sha256(json.dumps(calls).encode()).hexdigest()
    assert digest == ARM_B_PROMPTS_SHA256, digest


# ── knobs the proposer cannot edit ────────────────────────────────────────


@pytest.mark.parametrize("bad, message", [
    ("max_completion_tokens", "is frozen"),
    ("missing_skill", "no 'missing_skill'"),
])
def test_frozen_missing_or_non_text_knobs_are_refused_before_spending(bad, message):
    """Fails when a frozen or missing knob is accepted as a section."""
    with pytest.raises(ValueError, match=message):
        _proposer((*KNOBS[:3], bad)).check_knobs(_seed_state())


def test_an_open_non_text_knob_is_refused():
    state = _seed_state()
    state["knobs"]["data_parallelism"]["spec"] = {
        **state["knobs"]["data_parallelism"]["spec"], "frozen": False}
    with pytest.raises(ValueError, match="has kind 'int'"):
        _proposer(("compare__system_prompt", "data_parallelism")).check_knobs(state)
    _proposer(KNOBS).check_knobs(state)


def test_the_knobs_option_parses_and_refuses_bad_values():
    loop = SkillOptLoop.__new__(SkillOptLoop)
    loop.configure(knobs="compare__system_prompt, decide__system_prompt")
    assert loop.proposer.knobs == ("compare__system_prompt", "decide__system_prompt")
    assert loop.proposer.knob == "decide__system_prompt"
    loop.configure()
    assert loop.proposer.knobs == ("skill",)
    for bad in ("a,,b", "a,a", 3):
        with pytest.raises(TypeError):
            loop.configure(knobs=bad)
    with pytest.raises(TypeError, match="not both"):
        loop.configure(knob="skill", knobs="compare__system_prompt,decide__system_prompt")


def test_a_resume_with_other_knobs_is_refused():
    proposer = _proposer()
    saved = proposer.state_dict()
    with pytest.raises(ValueError, match="cannot resume"):
        SkillOptProposer().load_state_dict(saved)
    _proposer().load_state_dict(saved)
    with pytest.raises(ValueError, match="start a new run"):  # a state from before the record
        SkillOptProposer().load_state_dict({"meta_skill_text": "m"})


# ── livemath/skill_graph end to end, scripted ─────────────────────────────

#: An analyst's edit to text every step's system prompt shares, named to precision's
#: section; it must survive the merge and the rank and land in precision alone.
NAMED_EDIT = {"op": "replace", "target": "## Task Format",
              "content": "## Task Format (NAMED_EDIT)", "section": "precision__system_prompt",
              "source_type": "failure", "support_count": 1}

#: Epoch 1's merged patch: one edit per kind of destination, the named edit, two
#: refused before ranking (cross-section, unnamed shared target) and one ranked out.
EPOCH1_EDITS = [
    {"op": "replace", "target": "Compare all options before committing.",
     "content": "CMP_EDIT Compare all options before committing.",
     "source_type": "failure", "support_count": 2},
    {"op": "insert_after", "target": "## Hypotheses", "content": "- HYP_EDIT check every assumption.",
     "source_type": "failure", "support_count": 1},
    # compare's last words, the blank line and precision's SECTION line: refused.
    {"op": "replace", "target": "step decides.\n\n\n<!-- SECTION precision__system_prompt",
     "content": "SPLIT", "source_type": "failure", "support_count": 1},
    {"op": "append", "content": "- APPEND_EDIT end with the label.", "source_type": "success",
     "support_count": 1},
    NAMED_EDIT,
    # In all four system prompts and unnamed: refused, so it takes no rank slot.
    {"op": "replace", "target": "## Task Format", "content": "AMBIGUOUS",
     "source_type": "failure", "support_count": 1},
    # Ranked out (the edit budget is 4) once the two refused edits are dropped.
    {"op": "insert_after", "target": "## Theorem-Level Precision", "content": "- PREC_DROPPED",
     "source_type": "success", "support_count": 1},
]


class GraphClient(FakeClient):
    """The scripted model for both callers: the graph's agent steps (no call key;
    ``decide`` always answers B) and the optimizer (FakeClient's replies, keyed by
    the call in flight). Logs every step call's (node, system prompt)."""

    def __init__(self, responses: dict[tuple, str] | None = None, *, crash_at: tuple | None = None,
                 node: str = "", steps: list | None = None) -> None:
        super().__init__(responses)
        self.node = node
        self.steps: list[tuple[str, str]] = [] if steps is None else steps
        self.crash_at = crash_at

    def for_node(self, node_id: str, role: str = "") -> "GraphClient":
        child = GraphClient(self.responses, crash_at=self.crash_at, node=node_id, steps=self.steps)
        child.calls = self.calls
        return child

    def with_tool_policy(self, policy: str) -> "GraphClient":
        return self

    def with_wall(self, wall: Any) -> "GraphClient":
        return self

    def with_budget(self, budget: Any) -> "GraphClient":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:  # type: ignore[override]
        key = current_call_key()
        if key is None:
            self.steps.append((self.node, system))
            return "<answer>B</answer>" if self.node == "decide" else f"{self.node} notes"
        if self.crash_at is not None and tuple(key) == self.crash_at:
            raise ProcessKilled(f"model died at {self.crash_at}")
        return super().complete(prompt, system=system)


def _responses() -> dict[tuple, str]:
    return {
        ("analyst_failure", 1, 0, 0): json.dumps({"batch_size": 1, "failure_summary": [], "patch": {
            "reasoning": "a", "edits": [{k: v for k, v in NAMED_EDIT.items()
                                         if k not in ("source_type", "support_count")}]}}),
        ("merge_final", 1, 0, 0): json.dumps({"reasoning": "m", "edits": EPOCH1_EDITS}),
        ("rank", 1, 0, 0): json.dumps({"reasoning": "r", "selected_indices": [0, 1, 2, 3]}),
    }


def _graph_config(root: Path, **overrides: Any) -> LoopConfig:
    root.mkdir(parents=True, exist_ok=True)
    parts = root / "parts.json"
    parts.write_text(json.dumps({"train": TRAIN, "val": VAL}))
    base: dict[str, Any] = dict(
        epochs=10, batch_size=1, optimizer="random", seed=42, runs_root=root / "run",
        allowed_ops=("set_knob",), strategy="skillopt", parts_manifest=str(parts),
        max_evaluations=200,
    )
    base.update(overrides)
    return LoopConfig(**base)


def _knob_texts(candidate: Candidate) -> str:
    return "\n".join(str(candidate.state["knobs"][k]["value"]) for k in KNOBS)


@pytest.fixture
def graph_run(monkeypatch):  # type: ignore[no-untyped-def]
    """Run SkillOpt on livemath/skill_graph over its four knobs, scripted."""
    _patch_task(monkeypatch)
    # Val rewards the epoch-1 edits, so the gate accepts epoch 1 and rejects epoch 2.
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(
        lambda run, cand: 0.1 * _knob_texts(cand).count("_EDIT")))

    def run(root: Path, client: GraphClient, *, resume: bool = False,
            options: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
        loop = strategies.create(
            "skillopt", MODE, _graph_config(root, resume=resume),
            strategy_options={"max_epochs": 2, "knobs": ",".join(KNOBS), "edit_budget_max": 4,
                              "edit_budget_min": 4, **(options or {})},
            project=Path("."), model=client,
        )
        loop.run()
        return loop

    return run


def _trace(root: Path) -> list[dict[str, Any]]:
    path = root / "run" / "skillopt_trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _steps(trace: list[dict[str, Any]]) -> list[str]:
    return [e["step_type"] for e in trace]


def test_skill_graph_runs_skillopt_over_its_four_knobs(tmp_path, graph_run):
    """One SkillOpt run over the four system-prompt knobs: every phase runs once per
    epoch on the composite, the patch becomes set_knob patches on the right knobs
    (the cross-section and the unnamed shared-target edits refused before ranking,
    an analyst's named edit carried through merge and rank to its section alone),
    each edit reaches the step that reads its
    knob, the slow update lands at the end of decide__system_prompt, and no frozen knob
    moves. Fails when edits go to the wrong knob or the phases see one knob."""
    client = GraphClient(_responses())
    loop = graph_run(tmp_path, client)
    trace = _trace(tmp_path)
    strategy = loop._custom_strategy
    steps = _steps(trace)
    for step in ("analyst_failure", "analyst_success", "merge_final", "rank", "patch_applied",
                 "val_gate_accept", "val_gate_reject", "slow_update_placeholder",
                 "slow_update", "meta_skill"):
        assert step in steps, step

    # Every optimizer call saw the composite.
    analyst = next(c for c in client.calls if c["call_key"][0] == "analyst_failure")
    assert "## Current Skill\n# Skill sections\n" in analyst["prompt"]
    assert "<!-- SECTION hypotheses__system_prompt:" in analyst["prompt"]
    # The trajectories show which step wrote what.
    assert "[compare] compare notes" in analyst["prompt"]
    assert "[decide] <answer>B</answer>" in analyst["prompt"]

    # The section field is part of every step's edit format, and the named edit
    # carries its section from the analyst through the merge to the rank.
    for kind in ("analyst_failure", "analyst_success", "merge_final", "rank"):
        call = next(c for c in client.calls if c["call_key"][0] == kind)
        assert call["system"].endswith(prompts.SECTION_EDIT_NOTE), kind
    merge_call = next(c for c in client.calls if c["call_key"] == ("merge_final", 1, 0, 0))
    assert '"section": "precision__system_prompt"' in merge_call["prompt"]
    rank_call = next(c for c in client.calls if c["call_key"] == ("rank", 1, 0, 0))
    assert "section='precision__system_prompt'" in rank_call["prompt"]
    # The two refused edits were dropped before ranking: the ranker saw five.
    assert "[4] " in rank_call["prompt"] and "[5] " not in rank_call["prompt"]
    assert "content='SPLIT'" not in rank_call["prompt"]
    assert "content='AMBIGUOUS'" not in rank_call["prompt"]
    refused = next(e for e in trace if e["step_type"] == "section_refused" and e["epoch"] == 1)
    assert [(o["index"], o["status"]) for o in refused["outcomes"]] == [
        (2, CROSS_SECTION), (5, AMBIGUOUS_SECTION)]

    patch = next(e for e in trace if e["step_type"] == "patch_applied" and e["epoch"] == 1)
    outcomes = {o["index"]: (o["knob"], o["status"]) for o in patch["edit_report"]["outcomes"]}
    assert outcomes == {
        0: ("compare__system_prompt", "applied_replace"),
        1: ("hypotheses__system_prompt", "applied_insert_after"),
        2: ("decide__system_prompt", "applied_append"),
        3: ("precision__system_prompt", "applied_replace"),
    }
    assert patch["edit_report"]["knobs"] == [
        "compare__system_prompt", "precision__system_prompt", "hypotheses__system_prompt",
        "decide__system_prompt"]

    population = loop.population
    edits = [e for c in population for e in c.edits]
    assert edits and {e["op"] for e in edits} == {"set_knob"}
    assert {e["name"] for e in edits} <= set(KNOBS)
    seed = population[0].state["knobs"]
    for candidate in population:
        for name, entry in candidate.state["knobs"].items():
            if name not in KNOBS:
                assert entry == seed[name], name

    current = next(c for c in population if c.id == strategy.gate.current_id)
    knobs = current.state["knobs"]
    assert "CMP_EDIT" in knobs["compare__system_prompt"]["value"]
    assert "HYP_EDIT" in knobs["hypotheses__system_prompt"]["value"]
    assert "## Task Format (NAMED_EDIT)" in knobs["precision__system_prompt"]["value"]
    assert "PREC_DROPPED" not in knobs["precision__system_prompt"]["value"]
    for knob in ("compare__system_prompt", "hypotheses__system_prompt", "decide__system_prompt"):
        assert "NAMED_EDIT" not in knobs[knob]["value"], knob
    assert "AMBIGUOUS" not in _knob_texts(current)
    decide = knobs["decide__system_prompt"]["value"]
    assert "APPEND_EDIT" in decide
    assert decide.rstrip().endswith(
        f"{prompts.SLOW_UPDATE_START}\nUpdated guidance text.\n{prompts.SLOW_UPDATE_END}")
    assert "SPLIT" not in _knob_texts(current) and "PREC_DROPPED" not in _knob_texts(current)

    # Each edit reached the step whose system prompt reads its knob, and only it.
    epoch2 = [(node, system) for node, system in client.steps if "CMP_EDIT" in system
              or "HYP_EDIT" in system or "APPEND_EDIT" in system]
    assert {node for node, system in epoch2 if "CMP_EDIT" in system} == {"compare"}
    assert {node for node, system in epoch2 if "HYP_EDIT" in system} == {"hypotheses"}
    assert {node for node, system in epoch2 if "APPEND_EDIT" in system} == {"decide"}
    assert {node for node, system in client.steps if "NAMED_EDIT" in system} == {"precision"}
    assert strategy.strategy_state_dict()["meta_skill_text"] == "USE CONCISE EDITS"

    # Every candidate the run holds (the accepted patch included) is a child of one
    # it holds, so the lineage of the best reaches the seed.
    by_id = {c.id: c for c in population}
    assert any(e.get("patch") for c in population for e in c.edits[-1:])
    assert all(c.parent in by_id for c in population[1:])
    best = by_id[strategy.gate.best_id]
    lineage = loop._lineage(best)
    assert lineage[0]["id"] == population[0].id and lineage[-1]["id"] == best.id
    assert len(lineage) >= 2


def test_call_keys_stay_unique_and_replay_with_four_knobs(tmp_path, graph_run):
    """Every optimizer call has its own key across a two-epoch four-knob run, and
    feeding the recorded replies back by key reproduces every decision. Fails when
    a phase is run once per knob (repeated keys) or a reply is not replayable."""
    first = GraphClient(_responses())
    graph_run(tmp_path / "a", first)
    trace = _trace(tmp_path / "a")
    calls = [e for e in trace if e["type"] == "model_call"]
    keys = [tuple(e["call_key"]) for e in calls]
    assert len(keys) == len(set(keys)) == len(first.calls) > 0
    assert ("slow_update", 2, 0, 0) in keys and ("meta_skill", 2, 0, 0) in keys

    recorded = {tuple(e["call_key"]): e["raw_output"] for e in calls}
    replay = GraphClient(recorded)
    graph_run(tmp_path / "b", replay)
    strip = lambda t: [{k: v for k, v in e.items() if k != "timestamp"} for e in t]
    assert strip(_trace(tmp_path / "b")) == strip(trace)


@pytest.mark.parametrize("crash_at", [("merge_final", 2, 0, 0), ("slow_update", 2, 0, 0)])
def test_a_four_knob_run_killed_mid_epoch_resumes_where_it_stopped(tmp_path, graph_run, crash_at):
    """Killed at an epoch-2 optimizer call, the resumed run makes exactly the
    uninterrupted run's remaining calls, with the same composite prompts, and ends
    on the same candidate. Fails when resume rebuilds the epoch or loses a knob."""
    reference = GraphClient(_responses())
    ref_loop = graph_run(tmp_path / "ref", reference)

    crashing = GraphClient(_responses(), crash_at=crash_at)
    with pytest.raises(ProcessKilled):
        graph_run(tmp_path / "split", crashing)
    resumed = GraphClient(_responses())
    loop = graph_run(tmp_path / "split", resumed, resume=True)

    keys = [c["call_key"] for c in reference.calls]
    assert [c["call_key"] for c in crashing.calls] == keys[:keys.index(crash_at)]
    assert [c["call_key"] for c in resumed.calls] == keys[keys.index(crash_at):]
    assert [c["prompt"] for c in crashing.calls + resumed.calls] == [c["prompt"] for c in reference.calls]
    assert loop._custom_strategy.gate.current_id == ref_loop._custom_strategy.gate.current_id


def test_a_four_knob_run_resumed_before_the_val_gate_rebuilds_its_patch(tmp_path, graph_run,
                                                                        monkeypatch):
    """Killed at the epoch-1 val gate, after the four-knob patch, the resumed run
    rebuilds the same patched candidate (through run.attempt, knob by knob) and
    gates it. Fails when the rebuild applies the patch to one knob only."""
    from refactory_dsh.outerloop import Run

    original = Run.evaluate
    seen: dict[str, list[str]] = {"ref": [], "split": [], "resumed": []}
    label = {"run": "ref", "armed": False}

    def evaluate(self, candidate, *, part="train", items=None, seed=None):  # type: ignore[no-untyped-def]
        if part == "val":
            if label["armed"] and len(seen[label["run"]]) == 1:
                raise RuntimeError("process died at the val gate")
            seen[label["run"]].append(candidate.id)
        return original(self, candidate, part=part, items=items, seed=seed)

    monkeypatch.setattr(Run, "evaluate", evaluate)
    graph_run(tmp_path / "ref", GraphClient(_responses()))
    label.update(run="split", armed=True)
    with pytest.raises(RuntimeError, match="val gate"):
        graph_run(tmp_path / "split", GraphClient(_responses()))
    label.update(run="resumed", armed=False)
    graph_run(tmp_path / "split", GraphClient(_responses()), resume=True)
    assert seen["resumed"] == seen["ref"][1:]
