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
