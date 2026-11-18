"""The optimizer's own words are said once in a proposal request.

A proposal request shows what the optimizer wrote before -- a knob text it set, a patch, a
new step's prompt, its prediction -- in several places: the attempts list, the evidence
table, the last edit's child, its earlier reasoning, the settings. Built here from a
two-epoch history with a real loop (a ``set_knob`` rewrite in epoch 1, a structural child
adding a fork branch with its own prompt and system prompt in epoch 2), no sentence of
100+ characters that the optimizer wrote is said twice: each place that is not its home
names where it is instead.

The home of a knob's current text is "Each step's settings" (an exact edit target, a
patch quotes it), and the home of the last child's edits and prediction is "What the
last edit changed".
"""

from __future__ import annotations

import re

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import strategies
from refactory_dsh.loop import LoopConfig
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate
from refactory_dsh.tools.request_sizes import REPEAT_SENTENCE

#: The optimizer's own texts, each with sentences of 100+ characters.
KNOB_ADDED = ("Before you write a single note, restate the question in your own words and list "
              "every quantity it names together with its units and its constraints. Check each "
              "option against those constraints one at a time, and strike any option that "
              "violates one of them before comparing what is left.")
BRANCH_TEXT = ("Work the problem backwards from each option: assume the option is correct and "
               "derive what the question's givens would then have to be, and say which option "
               "is the only one consistent with every given in the question.")
BRANCH_SYSTEM = ("You are a careful verifier of mathematical answers who never trusts a single "
                 "line of reasoning and always checks a claimed answer by substituting it back "
                 "into the original conditions of the problem.")
PREDICTION = ("The backward check catches the items where the compare step picks an option "
              "that contradicts a given, so at least two of the currently failing train items "
              "should pass and none of the passing ones should break.")
RATIONALE = ("Most failures pick an option that violates a stated constraint, which a step "
             "that substitutes each option back into the givens would have caught early.")
FIRST_RATIONALE = ("The compare step lists options without checking them against the "
                   "question's constraints, so a rule to restate the constraints first should "
                   "cut the picks that break one.")
FIRST_FLOW = ("The compare step now restates every constraint of the question before it "
              "writes a note, so the later steps read options already checked against them.")
FIRST_PREDICTION = ("Items whose wrong pick broke a stated constraint should now pass, since "
                    "compare strikes such options before anything later reads its notes.")
FLOW = ("The fork now runs a third branch that substitutes each option back into the givens, "
        "and decide reads its verdict beside the precision and hypotheses notes.")


def _loop(tmp_path: Path, records: list[dict[str, Any]]) -> Any:
    loop = strategies.create("default", "livemath/skill_graph", LoopConfig(
        epochs=2, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path / "run",
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    knob = str(seed.state["knobs"]["compare_task"]["value"])
    first = {"op": "set_knob", "name": "compare_task", "value": f"{knob}\n\n{KNOB_ADDED}"}
    second = {"op": "branch", "fork": "step_fork", "kind": "agent", "text": BRANCH_TEXT,
              "system": BRANCH_SYSTEM}
    one = Candidate.from_state(nn_edits.apply(first, seed.state), parent=seed.id,
                               edits=[first])
    two = Candidate.from_state(nn_edits.apply(second, one.state), parent=one.id,
                               edits=[first, second])
    for epoch, (candidate, score) in enumerate(((seed, 0.5), (one, 0.55), (two, 0.6))):
        candidate.evaluated, candidate.score = True, score
        loop.population.append(candidate)
        loop._train_item_records[candidate.id] = records
        if epoch:
            loop.proposals[candidate.id] = {
                "epoch": epoch, "origin": "optimizer",
                "rationale": FIRST_RATIONALE if epoch == 1 else RATIONALE,
                "prediction": FIRST_PREDICTION if epoch == 1 else PREDICTION,
                "flow_change": FIRST_FLOW if epoch == 1 else FLOW}
    loop.backward()
    return loop


@pytest.fixture(params=["child", "parent"])
def request_text(request, tmp_path, monkeypatch) -> str:
    """The request editing the structural child itself (its texts are the current knobs),
    or editing its parent (the child is the last edit, its texts in no knob shown)."""
    from test_livemath_blocks import _patch_task
    from test_request_guardrails import _records

    _patch_task(monkeypatch)
    records = _records("livemath/skill_graph", tmp_path, runs=1)
    loop = _loop(tmp_path, records)
    if request.param == "parent":
        loop.strategy_decision["parent"] = loop.population[1].id
    return replace(loop._context(), item_records=records).render()


def test_the_fixture_shows_the_optimizer_s_texts(request_text):
    for text in (KNOB_ADDED, BRANCH_TEXT, BRANCH_SYSTEM, PREDICTION):
        assert text[:60] in request_text, text[:60]


def _own_sentences() -> list[str]:
    """The optimizer's own sentences of `request_sizes.REPEAT_SENTENCE`+ characters."""
    own = (KNOB_ADDED, BRANCH_TEXT, BRANCH_SYSTEM, PREDICTION, RATIONALE, FLOW,
           FIRST_RATIONALE, FIRST_PREDICTION, FIRST_FLOW)
    return [s for text in own for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s) >= REPEAT_SENTENCE]


def test_no_sentence_the_optimizer_wrote_is_said_twice(request_text):
    """Each of its long sentences appears once, counted by its first
    `request_sizes.REPEAT_SENTENCE` characters wherever they stand -- in a table cell, a
    JSON edit, a diff line, a fence -- so a shortened copy counts too."""
    folded = " ".join(request_text.split())
    said = {s[:60]: folded.count(s[:REPEAT_SENTENCE]) for s in _own_sentences()}
    assert said and all(n <= 1 for n in said.values()), said

