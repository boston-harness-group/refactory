"""The repeated-sentence guard (`request_sizes.repeated_sentences`) finds what it is
for: a long sentence said twice, however it is formatted, and nothing else."""

from __future__ import annotations

from refactory_dsh.tools.request_sizes import REPEAT_SENTENCE, repeated_sentences

LONG = ("The judge sees what the step read and wrote unless sees says otherwise, and its "
        "verdict decides whether the step runs again")


def test_a_sentence_in_a_table_cell_and_again_in_a_list_item_is_found():
    cell = LONG.replace("judge", "`judge`").replace("verdict", "**verdict**")
    text = f"| step | note |\n|---|---|\n| loop | {cell}. |\n\n- {LONG}.\n"
    assert repeated_sentences(text) == [LONG]


def test_twice_is_enough_and_once_is_not():
    assert repeated_sentences(f"{LONG}. Something else.\n{LONG}.") == [LONG]
    assert repeated_sentences(f"{LONG}.") == []


def test_a_long_sentence_whose_clauses_are_short_is_found_whole():
    clauses = ["the gate rule is said once under the score", "the counts once under items",
               "the score format once everywhere a score is printed"]
    sentence = "; ".join(clauses)
    assert all(len(c) < REPEAT_SENTENCE for c in clauses) and len(sentence) >= REPEAT_SENTENCE
    assert repeated_sentences(f"{sentence}.\n\n{sentence}.") == [sentence]


def test_a_long_clause_inside_two_different_sentences_is_found():
    clause = LONG
    found = repeated_sentences(f"First; {clause}.\n\nSecond; {clause}.")
    assert found == [clause]


def test_fenced_text_and_short_repeats_are_not_checked_by_decision():
    assert repeated_sentences(f"```\n{LONG}.\n```\n{LONG}.") == []
    short = "| solve | the step that answers |\n| check | the step that answers |"
    assert repeated_sentences(short) == []


def test_with_fenced_a_sentence_in_a_fence_and_an_indented_fence_is_found():
    """A step's instructions in its settings' fence and again inside an item's prompt,
    indented under the model client's wrapper, are the same sentence with ``fenced``."""
    text = f"```\n{LONG}.\n```\n\n```\n- User: ## Your Task\n  {LONG}.\n```\n"
    assert repeated_sentences(text, fenced=True) == [LONG]
    assert repeated_sentences(text) == []


def test_the_compute_policy_is_said_once():
    """The compute policy is said once, at the top: twice is a repeat like any other."""
    from refactory_dsh.optim.base import compute_policy

    policy = compute_policy(0.01).rstrip(".")
    assert len(policy) >= 100
    assert repeated_sentences(f"{policy}.") == []
    assert repeated_sentences(f"{policy}.\n\n{policy}.") == [policy]
