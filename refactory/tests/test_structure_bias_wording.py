"""Wording that biased the optimizer against structure, fixed (information only):

- the failure analysts' rule 7 asks for no change aimed at harness timeouts, and
  restricts no retry or loop aimed at a wrong answer;
- agreement is not waste: a check, verify or vote step confirming an answer, or parallel
  attempts agreeing, is not "repeated" or "restates_inputs", and no line says a step
  "adds a model call" (the compute policy prices calls);
- the integrity judge reads combining or selecting among the pipeline's own attempts
  by vote or check as legitimate;
- the memory note is evidence about the changes it names, never a rule against an
  untried design.
"""
from __future__ import annotations

from refactory_dsh.optim import analysts, feedback, integrity, memory


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_rule_seven_is_about_harness_timeouts_only():
    flat = _flat(analysts.FAILURE_SYSTEM)
    assert ("Do not ask for what the pipeline already does, or for retries or any other "
            "change aimed at harness timeouts") in flat
    assert "A retry, a check or a loop aimed at a wrong answer is a change like any other." in flat
    assert "for retries, or for a change aimed at timeouts" not in flat


def test_agreement_is_not_repetition_for_the_analysts():
    flat = _flat(analysts.FAILURE_SYSTEM)
    assert ("A check, verify or vote step confirming an answer, or parallel attempts "
            "agreeing, is not repetition") in flat
    assert "any step whose job was to produce something new" in flat
    assert "not a check, vote or agreeing attempt" in analysts._STEPS


def test_agreement_is_not_restating_for_the_step_review():
    sample = {"item": "i1", "input": "IN", "output": "OUT", "next_step": "", "next_output": ""}
    prompt = _flat(feedback.review_prompt({"step": "check", "candidate": "c", "kind": "loop/gate",
                                           "node_kind": "agent", "samples": [sample]}))
    assert ("A step whose job is to check, verify or vote (a judge, a verifier, a selector "
            "among attempts) and that confirms a right answer, or parallel attempts that "
            "agree, adds that confirmation") in prompt
    assert ("- restates_inputs: its job was to produce something new (an analysis, an "
            "attempt, a revision) and its output mostly repeats") in prompt
    lines = _flat("\n".join(feedback.step_review_lines(
        [{"step": "check", "candidate": "c", "kind": "other", "node_kind": "agent"}])))
    assert "adds a model call" not in lines
    assert "a check, verify or vote step that confirms an answer" in lines


def test_a_vote_or_selector_over_the_pipelines_own_attempts_is_legitimate():
    text = _flat(integrity._how_to_judge())
    assert ("combining or selecting among the pipeline's own attempts by vote or check") in text
    # The categories themselves (and so the veto logic) are unchanged.
    assert "hedging across several answers" in integrity.CATEGORIES["grader_exploit"]
    assert "picks, copies or defaults to an answer" in integrity.CATEGORIES["task_bypass"]


def test_the_note_is_weighed_as_evidence_never_a_rule_against_untried_designs():
    note = {"text": "NOTE", "epoch": 2}
    for shown in (memory.context_block(note), "\n".join(memory.request_lines(note))):
        flat = _flat(shown)
        assert "Prefer it when the current evidence is ambiguous" not in flat
        assert ("do not let it rule out a change nobody has tried, or a different design "
                "(other instructions, criteria, place or limit) of a change that failed") in flat
    rejected = "\n".join(memory.request_lines(
        {**note, "prev": "inc", "rejected": {"child": "rej"}}))
    assert "the unchanged incumbent `inc` and its rejected change `rej`" in rejected
