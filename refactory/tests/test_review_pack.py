"""The interactive review mode: the first mode with a gate a person answers.

Level 3 needs a mode that pauses for a person; every shipped mode was runnable end
to end without one. These hold the two properties the pause depends on: the IR says
who answers each gate, and a walker run stops at the person's gate instead of
inventing a verdict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import lowering, registry, steering

ROOT = Path(__file__).resolve().parent.parent.parent
MODE = "review_pack/review"


def _mode():
    return registry.resolve(ROOT, MODE)[1]()


def test_the_mode_builds_a_graph_that_validates():
    mode = _mode()
    import refactory_dsh.nn as nn

    issues = nn.validate_state(
        mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
    )
    assert issues == [], [str(issue) for issue in issues]


def test_the_ir_says_who_answers_each_gate():
    """`user` carries no command; `fn` carries one. Two answerers is one too many."""
    workflow = lowering.to_workflow(_mode().state_dict(), name="review")
    assert workflow.nodes["approve"].evaluator_type == "user"
    assert not workflow.nodes["approve"].evaluator_command, (
        "a person's gate was given a command, so a program could answer for them"
    )
    assert workflow.nodes["check"].evaluator_type == "fn"
    assert workflow.nodes["check"].evaluator_command


def test_the_author_note_is_the_prompt_the_person_is_shown():
    """A person asked to choose between labels needs the author's words, not `Gate approve`."""
    workflow = lowering.to_workflow(_mode().state_dict(), name="review")
    assert workflow.nodes["approve"].gate_prompt == "the person approves the draft, or halts the run"


def test_a_headless_run_refuses_at_the_persons_gate(tmp_path):
    """A walker run stops at a user gate with GateStop."""
    result = _mode().evaluate(tmp_path / "run")
    assert result.exit_ok is False
    assert "approve" in (result.error or ""), result.error


def test_the_persons_answer_completes_the_run_and_publishes(tmp_path):
    """The same run, resumed on the answer the ledger recorded."""
    work = tmp_path / "run"
    steering.Ledger.seed(work, "draft a note and let me approve it")
    steering.Ledger(work).verdict("approve", "proceed", approval=True)

    result = _mode().evaluate(work)
    assert result.exit_ok, result.error
    assert (work / "final.md").exists()
    assert "approved by the person" in (work / "final.md").read_text()
    assert result.score > 0


def test_a_halt_publishes_nothing_and_scores_zero(tmp_path):
    work = tmp_path / "run"
    steering.Ledger.seed(work, "draft a note and let me approve it")
    steering.Ledger(work).verdict("approve", "halt", approval=True)

    result = _mode().evaluate(work)
    assert result.exit_ok, result.error
    assert (work / "rejected.md").exists()
    assert not (work / "final.md").exists()
    assert result.score == 0


def test_the_fn_gate_refuses_a_thin_draft(tmp_path):
    """`rounds=1` makes a draft the deterministic gate will not send to a person.

    The reloop is bounded, so the run halts at the gate rather than reaching the
    person's gate; nothing is published and nothing is rejected.
    """
    work = tmp_path / "run"
    mode = _mode()
    state = mode.state_dict()
    state["knobs"]["rounds"]["value"] = 1
    mode = type(mode)(state=state)
    steering.Ledger.seed(work, "draft a note about circle packing")
    steering.Ledger(work).verdict("approve", "proceed", approval=True)

    result = mode.evaluate(work, task={"topic": "x"})
    assert result.exit_ok is False
    # SWEEP-8: a reloop cap is recorded as halted, not failed.
    assert result.halted is True, result.error
    assert not (work / "final.md").exists(), "a thin draft reached the person"
    assert not (work / "rejected.md").exists(), "a thin draft reached the person"


def test_the_fn_gate_says_how_thin_the_draft_is(tmp_path):
    """A refusal names the word count, so the redraft knows what is missing."""
    from refactory_dsh.nn import RunContext
    from refactory_packages.review_pack import ops

    ctx = RunContext(tmp_path, knobs={}, facts={}, task={})
    ctx.write_text(ops.DRAFT, "too short")
    verdict = ops.check(ctx)
    assert verdict == "reloop" and f"at least {ops.MIN_WORDS}" in verdict.reason
