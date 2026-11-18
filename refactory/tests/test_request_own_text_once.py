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


#: A long rewrite appended to a long knob (the seed's 780-character system prompt): the
#: added text is past any head a clipped listing would show.
LONG_RULES = " ".join(
    f"Rule {n}: before you commit to an option, check it against the {what} of the question, "
    f"and write down which given it would break if it were the answer, then move on."
    for n, what in enumerate(("units", "signs", "bounds", "equality cases", "domains",
                              "parities", "limits", "endpoints", "multiplicities",
                              "orientations"), 1))


def _loop(tmp_path: Path, records: list[dict[str, Any]], long_knob: bool = False) -> Any:
    loop = strategies.create("default", "livemath/skill_graph", LoopConfig(
        epochs=2, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path / "run",
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    knob = str(seed.state["knobs"]["compare_task"]["value"])
    first = {"op": "set_knob", "name": "compare_task", "value": f"{knob}\n\n{KNOB_ADDED}"}
    second = {"op": "branch", "fork": "step_fork", "kind": "agent", "text": BRANCH_TEXT,
              "system": BRANCH_SYSTEM}
    if long_knob:
        system = str(seed.state["knobs"]["compare__system_prompt"]["value"])
        second = {"op": "set_knob", "name": "compare__system_prompt",
                  "value": f"{system}\n\n{LONG_RULES}"}
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


@pytest.fixture(params=["child", "parent", "long knob"])
def request_text(request, tmp_path, monkeypatch) -> str:
    """The request editing the structural child itself (its texts are the current knobs),
    or editing its parent (the child is the last edit, its texts in no knob shown), or
    editing the parent of a child that appended over 1000 characters to a long knob."""
    from test_livemath_blocks import _patch_task
    from test_request_guardrails import _records

    _patch_task(monkeypatch)
    records = _records("livemath/skill_graph", tmp_path, runs=1)
    loop = _loop(tmp_path, records, long_knob=request.param == "long knob")
    if request.param != "child":
        loop.strategy_decision["parent"] = loop.population[1].id
    return replace(loop._context(), item_records=records).render()


def _shown(request_text: str) -> tuple[str, ...]:
    return ((KNOB_ADDED, LONG_RULES, PREDICTION) if LONG_RULES[:60] in request_text
            else (KNOB_ADDED, BRANCH_TEXT, BRANCH_SYSTEM, PREDICTION))


def test_every_text_the_optimizer_wrote_is_whole_once(request_text):
    """Each long text it wrote is shown whole, exactly once (a 1000+ character addition to
    a knob too, not cut after the knob's unchanged start)."""
    folded = " ".join(request_text.split())
    for text in _shown(request_text):
        assert folded.count(" ".join(text.split())) == 1, text[:60]


#: A pointer to a text: its head, its length and where it is whole.
POINTER = re.compile(r'\\?"([^"\\]{20,}?)…\\?" \((\d+) chars, whole '
                     r'(?:under \\?"([^"\\]+)\\?"|in `([^`]+)` under)')


def _section(text: str, heading: str) -> str:
    """The section under ``## heading``, to the next ``## `` heading outside a fence."""
    out, inside, fence = [], False, None
    for line in text.splitlines():
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
        elif fence is None and line.startswith("## "):
            inside = line.startswith("## " + heading)
        if inside:
            out.append(line)
    return "\n".join(out)


def test_every_pointer_s_target_holds_the_text_whole(request_text):
    """A pointer saying a text is whole in a section is true: the section holds it whole."""
    own = [" ".join(t.split()) for t in (KNOB_ADDED, LONG_RULES, BRANCH_TEXT, BRANCH_SYSTEM)]
    pointers = POINTER.findall(request_text)
    assert pointers, "the fixture's texts are pointed to"
    for head, chars, section, knob in pointers:
        text = next((t for t in own if t.startswith(head.strip())), None)
        if text is None:
            continue  # a seed text, not the optimizer's
        where = _section(request_text, section or "Each step's settings")
        assert text in " ".join(where.split()), (head, section or knob)


def _own_sentences() -> list[str]:
    """The optimizer's own sentences of `request_sizes.REPEAT_SENTENCE`+ characters."""
    own = (KNOB_ADDED, BRANCH_TEXT, BRANCH_SYSTEM, PREDICTION, RATIONALE, FLOW,
           FIRST_RATIONALE, FIRST_PREDICTION, FIRST_FLOW, LONG_RULES)
    return [s for text in own for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s) >= REPEAT_SENTENCE]


def test_no_sentence_the_optimizer_wrote_is_said_twice(request_text):
    """Each of its long sentences appears once, counted by its first
    `request_sizes.REPEAT_SENTENCE` characters wherever they stand -- in a table cell, a
    JSON edit, a diff line, a fence -- so a shortened copy counts too."""
    folded = " ".join(request_text.split())
    said = {s[:60]: folded.count(s[:REPEAT_SENTENCE]) for s in _own_sentences()}
    assert said and all(n <= 1 for n in said.values()), said



def test_knob_attribution_names_every_value_apart_and_never_merges_rows():
    """Two long values that are neither current nor the seed's, of one length and one
    head, each keep their row, named by the candidate first scored with it (or, without
    one, by a hash of the text)."""
    from refactory_dsh.optim.base import _trace_rows

    head = "x" * 200
    a, b, c = head + "a" * 100, head + "b" * 100, head + "c" * 100
    rows = dict(_trace_rows([{"knob_attribution": {"k": {"tried": {
        a: {"n": 3, "mean": 0.4, "max": 0.5, "first": "c1"},
        b: {"n": 2, "mean": 0.7, "max": 0.7, "first": "c2"},
        c: {"n": 1, "mean": 0.1, "max": 0.1},
        # A label two values share (here a stale 'first') still gives each its own row.
        head + "d" * 100: {"n": 1, "mean": 0.9, "max": 0.9, "first": "c1"}}}}}],
        current={"k": "now"}, seed={"k": "start"}))
    tried = rows["knob_attribution"]["k"]["tried"]
    assert len(tried) == 4 and sorted(row["mean"] for row in tried.values()) == [
        0.1, 0.4, 0.7, 0.9], tried
    assert tried["the text c1 was first scored with (300 chars)"] == {"n": 3, "mean": 0.4,
                                                                     "max": 0.5}
    assert tried["the text c2 was first scored with (300 chars)"]["mean"] == 0.7
    assert any(key.startswith("a text of 300 chars, #") for key in tried)


def test_the_loop_records_which_candidate_first_had_each_value(request_text):
    """A value neither current nor the seed's (the long-knob child's, its parent edited)
    is named by the candidate first scored with it, which the loop records."""
    if LONG_RULES[:60] in request_text:
        assert "was first scored with (2448 chars)" in request_text
    assert "the current text (" in request_text


def test_a_repair_loss_is_said_beside_the_refusals():
    from refactory_dsh.optim import ProposalContext

    loss = {"candidate": "c7", "dropped": "branch 'system' (412 chars)"}
    text = ProposalContext(rejections=["E-knob: no knob 'x'"], repair_losses=[loss]).render()
    refusals = text[text.index("## Why your last proposals were refused"):].split("\n## ", 1)[0]
    assert "`c7` is your refused proposal" in refusals and text.count("`c7` is your") == 1


# ── the memory request ─────────────────────────────────────────────────────


JUDGE_TEXT = ("You are a careful reviewer of the solver's answer. Check whether the answer "
              "falls into a maximalist trap, picking the strongest-sounding option without "
              "checking that each of its claims holds, and send it back with the reason.")


@pytest.mark.parametrize("rejected", [False, True])
def test_the_memory_request_says_a_change_s_text_once(rejected):
    """A change the memory request shows whole in its changes section ("What changed
    between them", or "What the rejected change changed") is pointed to from "What was
    tried this epoch", so the judge's text is said once."""
    from refactory_dsh.optim import memory
    from refactory_dsh.tools.request_sizes import repeated_sentences

    edit = {"op": "loop", "step": "solve", "judge": "agent", "max_iterations": 1,
            "text": JUDGE_TEXT}
    lineage = {"common": "p", "prev": [], "curr": [{"id": "c", "parent": "p", "edits": [edit],
                                                     "rationale": "R"}]}
    section = memory.changes_heading(memory.REJECTED_NOUN if rejected else None)
    entry = {"id": "c", "parent": "p", "round": 1, "kind": "loop/gate", "structural": True,
             "edits": memory.edits_text([edit]), "score": 0.5, "exit_ok": True,
             "evaluated": True, "fixed": 0, "broken": 2, "shared": 4, "gate": "rejected",
             "change": "loop step=solve", "change_key": "k"}
    tried = memory.tried_lines([entry], [], round_=1, current="p", gated=True,
                               tally=memory.change_tally_lines([entry], gated=True),
                               shown_in={"c": section})
    rec = {"id": "i", "passed": True, "score": 1.0, "prediction": "A", "exchanges": []}
    entries, _ = memory.pairs([rec], [{**rec, "passed": False, "score": 0.0}], ["i"])
    from refactory_dsh import registry
    from refactory_dsh.optim import analysts
    from refactory_dsh.optim.base import PIPELINE_VIEW_LEAD

    state = registry.resolve(Path("."), "livemath/solver")[1]().state_dict()
    child = nn_edits.apply(edit, state)
    pipes = [analysts.pipeline_text(memory.state_view(s), memory.PIPELINE_TEXT_CHARS)
             for s in (state, child)]
    curr = {"id": "c", "pipeline": pipes[1], "rollout": "R",
            **({"rejected": {"child": "c", "changed": 1, "shown": 1, "broken": 1, "fixed": 0,
                             "shared": 1, "others": []}} if rejected else {})}
    text, _ = memory.build_prompt(epoch=2, mode_name="m",
                                  prev={"id": "p", "pipeline": pipes[0], "rollout": "R"},
                                  curr=curr,
                                  changes=memory.changes_lines(lineage, homes={
                                      "curr": (memory.pipeline_homes(child), "the new pipeline")}),
                                  previous_note="",
                                  entries=entries, tried=tried)
    assert text.count(JUDGE_TEXT) == 1
    assert f'its edits are under "{section}" above' in text
    assert not [s for s in repeated_sentences(text) if "maximalist" in s or "reviewer" in s]
    # Both pipelines are shown; how a pipeline is drawn is said once (S1-s45 e2: the legend
    # twice was .115 of the request's repeated share).
    assert text.count(PIPELINE_VIEW_LEAD) == 1
    assert memory.SHARED_LINE_NOTE in text
    assert not [s for s in repeated_sentences(text) if "one line per step" in s]
