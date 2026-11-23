"""The wiring warnings an edit adds reach its proposer before its candidate runs.

The SpreadsheetBench V2 shakedown (2026-11-23): an ``insert`` with ``takes_over`` of a
loop's judge left the old judge running with its output unread (10 of 14 reloops
ignored). The warning (``W-unread-output``) was right, but it reached the proposer only in
the next round's ledger, a whole evaluation later; once it saw it, it fixed it at once.
Now ``Run.attempt`` shows the proposer the warnings its edit adds as soon as the edit
applies (``Loop._wiring_check``), in one slim call (`repair.wiring_request`): it changes
the wiring, or keeps the change and says why. Never a refusal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structure_fuzz as sf
from refactory_dsh.loop import Loop, LoopConfig, _new_wiring_warnings
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim.compose import PickedParent
from refactory_dsh.optim.policies import HillClimbSelector
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
from refactory_dsh.outerloop import Run

MODE = "circle_packing/adaptive"
#: A step whose output nothing reads: a new op after ``accept`` writing a file of its own.
ORPHAN = {"op": "insert_after", "target": "accept",
          "node": {"id": "scribe", "kind": "op", "call": "refactory_dsh.nn.loops:loop_end",
                   "writes": ["scribe.txt"], "score_effect": "none"}}
KNOB = {"op": "set_knob", "name": "budget", "value": 4}


def _judge_taken_over() -> tuple[dict, dict]:
    """The shakedown's shape on the fuzz seed: a ``loop`` with an agent judge, then a new
    step that takes the judge over and sees only the task and the draft."""
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    looped = nn_edits.apply({"op": "loop", "step": "draft", "judge": "agent",
                             "text": "Check the draft."}, state)
    judge = next(n for n in looped["graph"]["nodes"] if n.startswith("draft_check_judge"))
    taken = nn_edits.apply({"op": "insert", "after": judge, "kind": "agent",
                            "text": "Verify the check.", "takes_over": judge,
                            "sees": ["task", "draft"]}, looped)
    return looped, taken


def test_a_take_over_that_orphans_the_step_it_took_over_is_warned_and_says_so():
    looped, taken = _judge_taken_over()
    [warning] = _new_wiring_warnings(taken, looped)
    assert warning.startswith("W-unread-output [draft_check_judge]: node 'draft_check_judge' "
                              "writes 'draft_check_judge_own_draft_check_judge.txt'")
    # The take-over is named, with the two ways out.
    assert ("Its earlier output is now written by 'draft_check_judge_next', which took it "
            "over, while 'draft_check_judge' still runs: if the new step replaces it, remove "
            "'draft_check_judge'") in warning
    # Only what the edit adds: the same graph against itself, or a text change, adds none.
    assert _new_wiring_warnings(taken, taken) == []
    knob = next(name for name in looped["knobs"] if name.startswith("draft_check_judge"))
    retexted = nn_edits.apply({"op": "set_knob", "name": knob, "value": "Check it again."}, looped)
    assert _new_wiring_warnings(retexted, looped) == []


def test_a_take_over_whose_new_step_reads_the_old_output_is_not_warned():
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    looped = nn_edits.apply({"op": "loop", "step": "draft", "judge": "agent",
                             "text": "Check the draft."}, state)
    judge = next(n for n in looped["graph"]["nodes"] if n.startswith("draft_check_judge"))
    taken = nn_edits.apply({"op": "insert", "after": judge, "kind": "agent",
                            "text": "Verify the check.", "takes_over": judge}, looped)
    assert _new_wiring_warnings(taken, looped) == []


class _Answers:
    """A model client answering each call with the next scripted payload."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.payloads.pop(0) if self.payloads else {"edits": []})


def _loop(tmp_path: Path, optimizer: Optimizer) -> Loop:
    loop = Loop(MODE, LoopConfig(optimizer="random", runs_root=tmp_path), project=Path("."))
    loop.init()
    loop.optimizer = optimizer
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    return loop


def _trace(tmp_path: Path) -> list[dict]:
    path = tmp_path / "optimizer_trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _journal(loop: Loop, event: str) -> list[dict]:
    from refactory_dsh.loop import JOURNAL

    lines = (loop.root / JOURNAL).read_text().splitlines()
    return [r for r in map(json.loads, lines) if r.get("event") == event]


def test_the_proposer_sees_the_warning_when_its_edit_applies_and_its_fix_is_the_candidate(
        tmp_path):
    client = _Answers({"edits": [KNOB], "why": "the scribe was a mistake; a bigger budget instead"})
    loop = _loop(tmp_path, AutoresearchOptimizer(client))
    parent = loop.population[0]
    outcome = Run(loop).attempt(Proposal(edit=ORPHAN, parent=parent.id, rationale="note it"),
                                parent)
    assert isinstance(outcome, Candidate), outcome
    # The fix is the candidate: no scribe, the knob changed.
    assert "scribe" not in outcome.state["graph"]["nodes"]
    assert outcome.state["knobs"]["budget"]["value"] == 4
    assert any(n.startswith("wiring warnings shown before it ran (W-unread-output): the "
                            "proposer changed the wiring") for n in outcome.notes), outcome.notes
    # The one call carried the warning, the proposal as written and the keep option.
    [prompt] = client.prompts
    assert "# Check the wiring of your proposal" in prompt
    assert "- W-unread-output [scribe]: node 'scribe' writes 'scribe.txt'" in prompt
    assert '"id": "scribe"' in prompt and "answer with an empty `edits` list" in prompt
    kinds = [t["kind"] for t in _trace(tmp_path)]
    assert kinds == ["wiring_review", "wiring_outcome"], kinds
    [review] = _journal(loop, "wiring_review")
    assert (review["outcome"], review["became"], review["remaining"]) == ("fixed", outcome.id, [])


def test_a_proposer_that_keeps_the_wiring_keeps_its_candidate_with_its_reason(tmp_path):
    client = _Answers({"edits": [], "why": "scribe.txt is read by a later op that opens it"})
    loop = _loop(tmp_path, AutoresearchOptimizer(client))
    parent = loop.population[0]
    outcome = Run(loop).attempt(Proposal(edit=ORPHAN, parent=parent.id), parent)
    assert isinstance(outcome, Candidate) and "scribe" in outcome.state["graph"]["nodes"]
    assert any("the proposer kept the change as it is; it said: scribe.txt is read by a later "
               "op" in n for n in outcome.notes), outcome.notes
    [review] = _journal(loop, "wiring_review")
    assert review["outcome"] == "kept" and review["remaining"] == review["warnings"]


def test_a_refused_fix_keeps_the_proposal_as_written(tmp_path):
    client = _Answers({"edits": [{"op": "remove_node", "id": "no_such_step"}], "why": "x"})
    loop = _loop(tmp_path, AutoresearchOptimizer(client))
    parent = loop.population[0]
    outcome = Run(loop).attempt(Proposal(edit=ORPHAN, parent=parent.id), parent)
    assert isinstance(outcome, Candidate) and "scribe" in outcome.state["graph"]["nodes"]
    [review] = _journal(loop, "wiring_review")
    assert review["outcome"] == "fix_refused" and review["refused"]


def test_an_edit_that_adds_no_warning_is_not_asked_about(tmp_path):
    client = _Answers()
    loop = _loop(tmp_path, AutoresearchOptimizer(client))
    parent = loop.population[0]
    outcome = Run(loop).attempt(Proposal(edit=KNOB, parent=parent.id), parent)
    assert isinstance(outcome, Candidate)
    assert client.prompts == [] and _trace(tmp_path) == []
    assert not any(n.startswith("wiring warnings") for n in outcome.notes)


def test_an_optimizer_that_asks_no_model_keeps_the_change_and_it_is_still_said(tmp_path):
    loop = Loop(MODE, LoopConfig(optimizer="random", runs_root=tmp_path), project=Path("."))
    loop.init()
    parent = loop.population[0]
    outcome = Run(loop).attempt(Proposal(edit=ORPHAN, parent=parent.id), parent)
    assert isinstance(outcome, Candidate)
    assert "wiring warnings shown before it ran (W-unread-output): the proposer kept the change " \
           "as it is" in outcome.notes


def test_a_parent_picker_hands_the_review_to_its_proposer(tmp_path):
    client = _Answers({"edits": [KNOB], "why": "fixed"})
    pair = PickedParent(HillClimbSelector(seed=0), AutoresearchOptimizer(client))
    loop = _loop(tmp_path, pair)
    parent = loop.population[0]
    outcome = Run(loop).attempt(Proposal(edit=ORPHAN, parent=parent.id), parent)
    assert isinstance(outcome, Candidate) and "scribe" not in outcome.state["graph"]["nodes"]
    assert len(client.prompts) == 1


def test_a_resumed_run_does_not_ask_about_the_same_edit_again(tmp_path):
    client = _Answers({"edits": [], "why": "meant"})
    loop = _loop(tmp_path, AutoresearchOptimizer(client))
    parent = loop.population[0]
    first = Run(loop).attempt(Proposal(edit=ORPHAN, parent=parent.id), parent)
    assert isinstance(first, Candidate) and len(client.prompts) == 1
    again = Loop(MODE, LoopConfig(optimizer="random", runs_root=tmp_path, resume=True),
                 project=Path("."))
    again.init()
    assert first.id in again._wiring_reviews
    other = _Answers({"edits": [], "why": "meant"})
    again.optimizer = AutoresearchOptimizer(other)
    Run(again).attempt(Proposal(edit=ORPHAN, parent=parent.id), again.population[0])
    assert other.prompts == []


class _Orphaning(Optimizer):
    """Proposes the orphaning edit once; shown its warning, answers with the knob instead."""

    name = "orphaning"

    def __init__(self) -> None:
        super().__init__(k=1)
        self.shown: list[list[str]] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        if ctx.epoch != 1:
            return []
        return [Proposal(edit=ORPHAN, parent=parents[0].id, rationale="note it")]

    def review_wiring(self, ctx, proposal, warnings, parent):  # type: ignore[no-untyped-def]
        self.shown.append(list(warnings))
        self.last_wiring_answer = "a bigger budget instead"
        return Proposal(edit=KNOB, parent=parent.id, rationale=proposal.rationale)


def test_in_a_loop_round_the_fixed_candidate_is_the_one_evaluated(tmp_path):
    """The same round: the warning is shown before the candidate is evaluated, and what
    is evaluated is the proposer's fix, never the orphaning edit."""
    loop = Loop(MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=5,
                                 runs_root=tmp_path), project=Path("."))
    loop.optimizer = optimizer = _Orphaning()
    loop.run()
    [shown] = optimizer.shown
    assert shown[0].startswith("W-unread-output [scribe]")
    children = [c for c in loop.population if c.parent is not None and c.evaluated]
    assert children and all("scribe" not in c.state["graph"]["nodes"] for c in children)
    assert any(c.state["knobs"]["budget"]["value"] == 4 for c in children)
    from refactory_dsh.loop import JOURNAL

    events = [json.loads(line)["event"] for line in (loop.root / JOURNAL).read_text().splitlines()]
    review_at = events.index("wiring_review")
    assert "score" in events[review_at:], "the fix was evaluated after its review"
