"""The per-proposal cap counts changes (`optim.edit_count`): a structural change and
the set_knob edits that make the steps reading its output use it count as one, decided
from the graph after the proposal is applied; unrelated text tunes count one each; and
a raw-edit ceiling (`MAX_RAW_EDITS_PER_PROPOSAL`) guards against runaway bundles."""
from __future__ import annotations

from typing import Any

import structure_fuzz as sf
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim.base import (MAX_EDITS_PER_PROPOSAL, MAX_RAW_EDITS_PER_PROPOSAL,
                                      Proposal, proposal_refusal)
from refactory_dsh.optim.edit_count import count, owners, readers

BRANCH = {"op": "branch", "fork": "step_fork", "kind": "agent",
          "text": "Check the units of each choice."}
INSERT = {"op": "insert", "after": "hypotheses", "kind": "agent",
          "text": "List the hypotheses each option needs."}


def _s4() -> dict[str, Any]:
    return sf.Applier(sf.seeds()["livemath_skill_graph"]).seed_state


def _tune(state: dict[str, Any], edit: dict[str, Any], line: str) -> dict[str, Any]:
    """A set_knob of the knob of the step that reads ``edit``'s new step, found from the
    graph after it is applied, adding ``line`` (which names no new step or output)."""
    out = nn_edits.apply(edit, state)
    new = set(out["graph"]["nodes"]) - set(state["graph"]["nodes"])
    reader = sorted(readers(out, new))[0]
    knob = next(name for name in out["knobs"] if reader in owners(out, name)
                and isinstance(out["knobs"][name].get("value"), str))
    return {"op": "set_knob", "name": knob, "value": out["knobs"][knob]["value"] + "\n" + line}


def _unrelated(name: str, state: dict[str, Any]) -> dict[str, Any]:
    return {"op": "set_knob", "name": name,
            "value": str(state["knobs"][name]["value"]) + "\nBe precise."}


def test_a_branch_and_the_tuning_of_its_reader_are_one_change():
    """S4: a new branch, `decide_task` tuned to use it (its op reads the branch's fact),
    and one unrelated prompt tune: 2 changes."""
    state = _s4()
    tune = _tune(state, BRANCH, "Give the units check the most weight.")
    assert tune["name"] == "decide_task"
    tally = count([BRANCH, tune, _unrelated("compare_task", state)], state)
    assert tally["logical"] == 2 and tally["groups"] == [[0, 1], [2]]


def test_two_structural_changes_with_their_readers_and_two_tunes_fit_the_cap():
    state = _s4()
    after_branch = nn_edits.apply(BRANCH, state)
    edits = [BRANCH, _tune(state, BRANCH, "Weigh the units check."),
             INSERT, _tune(after_branch, INSERT, "Use the hypotheses list."),
             _unrelated("compare__system_prompt", state),
             _unrelated("precision__system_prompt", state)]
    assert count(edits, state)["logical"] == 4 == MAX_EDITS_PER_PROPOSAL
    assert proposal_refusal(Proposal(edit={"edits": edits}), max_edits=MAX_EDITS_PER_PROPOSAL,
                            parent_state=state) == []


def test_five_unrelated_text_tunes_are_still_refused():
    state = _s4()
    names = ["compare_task", "compare__system_prompt", "precision__system_prompt",
             "hypotheses__system_prompt", "decide__system_prompt"]
    edits = [_unrelated(name, state) for name in names]
    assert count(edits, state)["logical"] == 5
    (refusal,) = proposal_refusal(Proposal(edit={"edits": edits}),
                                  max_edits=MAX_EDITS_PER_PROPOSAL, parent_state=state)
    assert refusal.startswith("E-too-many-edits: the proposal makes 5 changes")
    assert "The limit counts changes, not edits" in refusal


def test_no_proposal_carries_more_than_the_raw_ceiling():
    state = _s4()
    tune = _tune(state, BRANCH, "Weigh the units check.")
    edits = [BRANCH, *[{**tune, "value": tune["value"] + f"\n{n}"}
                       for n in range(MAX_RAW_EDITS_PER_PROPOSAL)]]
    assert count(edits, state)["logical"] == 1, "every tune folds into the branch"
    (refusal,) = proposal_refusal(Proposal(edit={"edits": edits}),
                                  max_edits=MAX_EDITS_PER_PROPOSAL, parent_state=state)
    assert f"no proposal may carry more than {MAX_RAW_EDITS_PER_PROPOSAL}" in refusal
