"""What the model proposer learns across a run (`optim.feedback`), each behind a switch.

- ``run_ledger``: "Every change so far, by kind" lists every evaluated candidate of the
  run (not only the recent ones), with the kind of change it made, train fixed/broken
  against its parent and the gate's word, after a tally per kind; bounded.
- ``lessons``: the answer's optional ``lessons`` are kept, journaled, shown back as "Your
  lessons so far" and survive a resume; an answer without them keeps the old ones.
- ``step_review``: each step the last round added gets its train counts and one review
  call (``step_review``), cached, whose failure is shown and never fatal.
- Readers' success patterns get their own heading and room.
- ``structure_guidance`` (off by default) adds what makes a new step pay off.

No new section shows a validation item or score. Each test fails when the behaviour it
names is reverted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.optim import ProposalContext, feedback
from refactory_dsh.optim.base import Candidate, Proposal
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
from refactory_dsh.outerloop import ValidationGate

from test_gate_aware_context import Recording, _passes_by_level, _val
from test_skillopt import TEST_IDS, VAL_IDS, _patch_mode
from test_val_gate import LevelProbe, _level, _loop


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


# ── change kinds ────────────────────────────────────────────────────────────


AGENT = {"id": "think", "kind": "agent", "writes": ["think.txt"], "prompt_template": "p"}
CODE = {"id": "count", "kind": "op", "call": "refactory_dsh.snippet:run", "code": "def main(i): ..."}


@pytest.mark.parametrize("edits,kind", [
    ([{"op": "set_knob", "name": "a__task", "value": "x"}], "prompt/setting"),
    ([{"op": "add_knob", "name": "k", "spec": {}}], "prompt/setting"),
    ([{"op": "insert_after", "target": "a", "node": AGENT}], "added agent step"),
    ([{"op": "insert_after", "target": "a", "node": AGENT},
      {"op": "set_knob", "name": "think__task", "value": "x"}], "added agent step"),
    ([{"op": "insert_before", "target": "a", "node": CODE}], "added code step"),
    ([{"op": "insert_after", "target": "a", "node": AGENT},
      {"op": "insert_after", "target": "think", "node": CODE}], "added code step"),
    ([{"op": "split", "target": "a", "into": [AGENT, {**AGENT, "id": "b"}]}], "split"),
    ([{"op": "loop", "target": "a", "gate": {"id": "g", "kind": "gate"}}], "loop/gate"),
    ([{"op": "insert_after", "target": "a", "node": {"id": "g", "kind": "gate"}}], "loop/gate"),
    ([{"op": "parallel", "target": "a", "branches": [AGENT]}], "parallel"),
    ([{"op": "bypass", "target": "a"}], "removal/rewire"),
    ([{"op": "add_edge", "source": "a", "target": "b"}], "removal/rewire"),
    ([{"op": "reorder", "first": "a", "second": "b"}], "removal/rewire"),
    ([{"op": "insert_after", "target": "a", "node": {"id": "x", "kind": "data"}}], "other"),
])
def test_a_change_is_filed_by_its_ops_and_the_kinds_of_node_it_adds(edits, kind):
    assert feedback.change_kind(edits) == kind


def test_a_duplicate_is_filed_by_the_kind_of_the_node_it_copies():
    # The copy is looked up by its new id (the target need not be in the state shown).
    state = {"graph": {"nodes": {"a2": {**AGENT, "id": "a2"}}}}
    assert feedback.change_kind([{"op": "duplicate", "target": "a", "new_id": "a2"}],
                                state) == "added agent step"


def test_what_a_change_touched_names_knobs_and_steps():
    edits = [{"op": "split", "target": "decide", "into": [AGENT, {**AGENT, "id": "final"}]},
             {"op": "set_knob", "name": "final__task", "value": "x"}]
    assert feedback.touched(edits) == ["think", "final", "decide", "final__task"]


# ── the run ledger ──────────────────────────────────────────────────────────


#: Eight rounds: more candidates than "Your earlier reasoning" keeps (REASONING_ENTRIES).
PLAN = [[(5, 7)], [(7, 6)], [(7, 8)], [(8, 4)], [(8, 9)], [(9, 3)], [(9, 2)], [(9, 6)], []]


def _ledger_run(tmp_path: Path, monkeypatch: Any, **overrides: Any):
    from refactory_dsh.loop import Loop

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    proposer = Recording(PLAN)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": len(PLAN), "val_gate": True,
                                  "max_evaluations": 200, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def _no_val(text: str, loop: Any) -> None:
    """No validation (or test) item id and no val score appears in ``text``."""
    for item in (*VAL_IDS, *TEST_IDS):
        assert f"Item {item}" not in text and f"| {item} |" not in text and f" {item}," not in text
    for score in loop._gate().cache.values():
        assert f"{score:.6g}" not in text and str(score) not in text, score


def test_the_ledger_lists_every_candidate_of_the_run_with_a_tally(tmp_path, monkeypatch):
    from refactory_dsh.optim.base import REASONING_ENTRIES

    loop, proposer = _ledger_run(tmp_path, monkeypatch)
    ctx = proposer.contexts[-1]
    made = [c for c in loop.population if c.parent and c.evaluated]
    assert len(made) > REASONING_ENTRIES, "setup: more than the recent candidates"
    assert [entry["id"] for entry in ctx.ledger] == [c.id for c in made]
    text = ctx.render()
    section = _section(text, "## Every change so far, by kind")
    assert all(f"| {c.id} |" in section for c in made), "every candidate, not only recent"
    kept = sum(1 for e in ctx.ledger if e["gate"] in feedback.KEPT_WORDS)
    nets = [e["fixed"] - e["broken"] for e in ctx.ledger if e["fixed"] is not None]
    assert (f"- prompt/setting: tried {len(made)}, kept {kept}, best train fixed-broken "
            f"{max(nets):+d}") in section
    ids = {_level(c): c.id for c in loop.population}
    row = next(line for line in section.splitlines() if line.startswith(f"| {ids[4]} |"))
    # 4 (from 8) breaks t3-t6 against 8: passes t1-t2 only.
    assert "| prompt/setting | level | 0/4 of 8 | rejected |" in row
    _no_val(section, loop)


def test_without_the_ledger_switch_there_is_no_ledger(tmp_path, monkeypatch):
    _loop_, proposer = _ledger_run(tmp_path, monkeypatch, run_ledger=False)
    ctx = proposer.contexts[-1]
    assert ctx.ledger == []
    assert "## Every change so far, by kind" not in ctx.render()


def _entry(index: int, kind: str = "prompt/setting", gate: str = "rejected",
           rationale: str = "R" * 1000, touched: list[str] | None = None) -> dict[str, Any]:
    return {"id": f"c{index:04d}", "round": index, "parent": "p", "kind": kind,
            "touched": touched or ["decide__task"], "fixed": 1, "broken": 2, "shared": 30,
            "gate": gate, "rationale": rationale}


def test_the_ledger_keeps_the_newest_rows_and_counts_the_older_in_the_tally():
    entries = [_entry(i, kind="added agent step" if i % 2 else "prompt/setting",
                      gate="kept" if i % 10 == 0 else "rejected", rationale="short")
               for i in range(300)]
    lines = feedback.ledger_lines(entries)
    text = "\n".join(lines)
    rows = [line for line in lines[-1].splitlines() if line.startswith("| c")]
    assert len(rows) == feedback.LEDGER_ROWS
    assert rows[-1].startswith("| c0299 |"), "the newest are kept"
    assert "- prompt/setting: tried 150, kept 30, best train fixed-broken -1" in text
    assert "- added agent step: tried 150, kept 0, best train fixed-broken -1" in text
    assert f"The {300 - len(rows)} oldest are counted in the tally above only" in text


def test_the_ledger_table_stays_within_its_characters():
    entries = [_entry(i, touched=[f"{'k' * 80}_{n}" for n in range(6)]) for i in range(300)]
    lines = feedback.ledger_lines(entries)
    rows = [line for line in lines[-1].splitlines() if line.startswith("| c")]
    assert len(lines[-1]) <= feedback.LEDGER_CHARS < len(rows[0]) * feedback.LEDGER_ROWS
    assert rows[-1].startswith("| c0299 |") and "+2 more" in rows[-1]
    assert "R" * (feedback.LEDGER_RATIONALE_CHARS + 1) not in "\n".join(lines)


def test_without_the_gate_the_ledger_has_no_gate_column():
    lines = feedback.ledger_lines([{**_entry(1), "gate": None}])
    assert "| gate |" not in lines[-1] and "kept" not in "\n".join(lines)


# ── lessons ─────────────────────────────────────────────────────────────────


class Lessons(Recording):
    """Writes lessons in rounds 1 and 3 (round 2's answer carries none)."""

    WRITES = {1: "LESSON-ONE: prompt moves on level help", 3: "LESSON-THREE: revised"}

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        out = super().propose(ctx, n, parents)
        self.last_lessons = self.WRITES.get(len(self.contexts), "")
        return out


def _lessons_run(tmp_path: Path, monkeypatch: Any, plan: list, **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    proposer = Lessons(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": len(plan), "val_gate": True, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def test_lessons_are_kept_shown_back_and_kept_when_an_answer_has_none(tmp_path, monkeypatch):
    loop, proposer = _lessons_run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 9)], []])
    first, second, third, fourth = (c.render() for c in proposer.contexts[:4])
    assert '"lessons": "<optional, short: what the results so far teach' in first
    assert "## Your lessons so far" not in first, "nothing written yet"
    assert "LESSON-ONE" in _section(second, "## Your lessons so far")
    assert "in epoch 1 (generation" in second
    assert "LESSON-ONE" in third, "round 2 wrote none: the old lessons stay"
    assert "LESSON-THREE" in fourth and "LESSON-ONE" not in fourth
    events = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    written = [e["text"] for e in events if e.get("event") == "optimizer_lessons"]
    assert written == [Lessons.WRITES[1], Lessons.WRITES[3]]
    _no_val(_section(fourth, "## Your lessons so far"), loop)


def test_lessons_survive_a_resume(tmp_path, monkeypatch):
    _lessons_run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 9)]])
    again = Recording([[]])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=4, val_gate=True, resume=True)
        resumed.optimizer = again
        resumed.run()
    assert resumed._lessons["text"] == Lessons.WRITES[3]
    assert "LESSON-THREE" in _section(again.contexts[0].render(), "## Your lessons so far")


def test_without_the_lessons_switch_nothing_is_asked_kept_or_shown(tmp_path, monkeypatch):
    loop, proposer = _lessons_run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], []], lessons=False)
    text = proposer.contexts[1].render()
    assert '"lessons"' not in text and "## Your lessons so far" not in text
    assert "optimizer_lessons" not in (loop.root / "journal.jsonl").read_text()


class _Answering:
    """A proposal client answering with ``answer``; step reviews with ``review``."""

    def __init__(self, answer: dict[str, Any], review: Any = None) -> None:
        self.answer, self.review = answer, review
        self.proposals: list[str] = []
        self.reviews: list[str] = []
        self.lock = __import__("threading").Lock()

    def complete(self, prompt: str, *, system: str = "") -> str:
        if system == feedback.STEP_REVIEW_SYSTEM:
            with self.lock:
                self.reviews.append(prompt)
            if isinstance(self.review, Exception):
                raise self.review
            return self.review if isinstance(self.review, str) else json.dumps(self.review)
        self.proposals.append(prompt)
        return json.dumps(self.answer)


def test_the_proposer_reads_lessons_from_its_answer_only_when_asked():
    long = "L" * 5000
    optimizer = AutoresearchOptimizer(_Answering({"proposals": [], "lessons": long}))
    optimizer.propose(ProposalContext(ask_lessons=True), 1, [])
    assert optimizer.last_lessons == "L" * (feedback.LESSONS_CHARS - 1) + "…"
    listed = AutoresearchOptimizer(_Answering({"proposals": [], "lessons": ["a", "b"]}))
    listed.propose(ProposalContext(ask_lessons=True), 1, [])
    assert listed.last_lessons == "- a\n- b"
    unasked = AutoresearchOptimizer(_Answering({"proposals": [], "lessons": "x"}))
    unasked.propose(ProposalContext(ask_lessons=False), 1, [])
    assert unasked.last_lessons == ""


def test_a_parent_picker_hands_its_proposers_lessons_through():
    from refactory_dsh.optim import compose, policies

    proposer = AutoresearchOptimizer(_Answering({"proposals": []}))
    paired = compose.PickedParent(policies.HillClimbSelector(), proposer)
    proposer.last_lessons = "FROM-PROPOSER"
    assert paired.last_lessons == "FROM-PROPOSER"
    paired.last_lessons = ""
    assert proposer.last_lessons == ""
    proposer.last_reply_problem = "BROKEN"
    assert paired.last_reply_problem == "BROKEN", "the loop reads why an answer was unread"
    paired.last_reply_problem = ""
    assert proposer.last_reply_problem == ""


def test_lessons_shown_are_bounded():
    lines = feedback.lessons_lines({"text": "x" * 9000, "epoch": 2, "generation": 3})
    assert len(lines[-1]) == feedback.LESSONS_CHARS


# ── added steps ─────────────────────────────────────────────────────────────


def _rec(item: str, passed: bool, steps: dict[str, str],
         prediction: str | None = None) -> dict[str, Any]:
    return {"id": item, "passed": passed, "score": float(passed), "prediction": prediction,
            "exchanges": [{"node": node, "prompt": f"GIVEN {node} {item}", "response": text}
                          for node, text in steps.items()]}


PARENT_STATE = {"graph": {"nodes": {"a": {"id": "a", "kind": "agent"},
                                    "decide": {"id": "decide", "kind": "agent"}}}}
CHILD_STATE = {"graph": {"nodes": {"a": {"id": "a", "kind": "agent"},
                                   "fork1": {"id": "fork1", "kind": "fork"},
                                   "think": {"id": "think", "kind": "agent"},
                                   "join1": {"id": "join1", "kind": "join"},
                                   "check": {"id": "check", "kind": "op", "code": "x"},
                                   "decide": {"id": "decide", "kind": "agent"}}}}


#: The answers the grader read. The child's differ from the parent's on t4 and t5 only
#: (t0 differs in case and spacing alone); its reasoning text differs on every item.
PARENT_ANSWERS = ["A", "B", "C", "D", "E", "A"]
CHILD_ANSWERS = [" a ", "B", "C", "D", "b", "B"]


def _pair_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parent = [_rec(f"t{i}", i < 3, {"a": "notes", "decide": f"Reasoning {i}"},
                   PARENT_ANSWERS[i]) for i in range(6)]
    child = [_rec(f"t{i}", i in (1, 2, 4), {"a": "notes", "think": f"THINK {i}",
                                            "decide": f"Reasoning {i}, reconsidered"},
                  CHILD_ANSWERS[i]) for i in range(6)]
    return parent, child


def test_added_steps_are_found_by_graph_difference_without_plumbing():
    assert feedback.added_steps(PARENT_STATE, CHILD_STATE) == [("think", "agent"),
                                                                ("check", "code")]


def test_an_added_steps_evidence_is_train_counts_and_samples():
    parent, child = _pair_records()
    evidence = feedback.step_evidence("think", parent_records=parent, child_records=child)
    assert (evidence["shared"], evidence["fixed"], evidence["broken"]) == (6, 1, 1)
    assert (evidence["compared"], evidence["final_changed"]) == (6, 2), "answers, not text"
    assert (evidence["text_compared"], evidence["text_changed"]) == (0, 0)
    assert evidence["ran"] == 6 and evidence["median_s"] is None
    samples = evidence["samples"]
    assert len(samples) == feedback.STEP_REVIEW_ITEMS
    assert [s["item"] for s in samples] == ["t4", "t0", "t1", "t2"], "changed outcomes first"
    assert samples[0] == {"item": "t4", "input": "GIVEN think t4", "output": "THINK 4",
                          "next_step": "decide", "next_output": "Reasoning 4, reconsidered"}
    code = feedback.step_evidence("check", parent_records=parent, child_records=child)
    assert code["ran"] == 0 and code["samples"] == []


def test_without_a_recorded_answer_the_output_text_is_compared_and_said():
    """t0 and t1 recorded no answer: their last outputs are compared as text (spacing
    alone is no change), counted apart from t2's answer, and the line says which."""
    parent = [_rec("t0", True, {"decide": "Answer:  1\n"}),
              _rec("t1", True, {"decide": "one"}), _rec("t2", True, {"decide": "r"}, "A")]
    child = [_rec("t0", True, {"x": "y", "decide": "Answer: 1"}),
             _rec("t1", True, {"x": "y", "decide": "two"}),
             _rec("t2", True, {"x": "y", "decide": "s"}, "B")]
    evidence = feedback.step_evidence("x", parent_records=parent, child_records=child)
    assert (evidence["compared"], evidence["final_changed"]) == (1, 1)
    assert (evidence["text_compared"], evidence["text_changed"]) == (2, 1)
    line = feedback.step_review_lines([{**evidence, "candidate": "c", "kind": "other"}])[-1]
    assert "final answer changed on 1/1 (100%; the answer the grader read)" in line
    assert ("final output text changed on 1/2 (50%; items that recorded no answer, so the "
            "last step's output text was compared)") in line
    only_text = feedback.step_evidence("x", parent_records=parent[:2], child_records=child[:2])
    text = feedback.step_review_lines([{**only_text, "candidate": "c", "kind": "other"}])[-1]
    assert "final answer changed" not in text
    assert "(50%; no answer recorded, so the last step's output text was compared)" in text


@pytest.mark.parametrize("payload,verdict", [
    ({"verdict": "adds_new_work", "example": "e", "note": "n"}, "adds_new_work"),
    ({"verdict": "Restates inputs"}, "restates_inputs"),
    ({"verdict": "contradicts-inputs"}, "contradicts_inputs"),
    ({"verdict": "no_effect", "example": "q" * 900}, "no_effect"),
])
def test_a_review_verdict_is_read(payload, verdict):
    found = feedback.parse_review(payload)
    assert found["verdict"] == verdict
    assert len(found["example"]) <= feedback.STEP_REVIEW_QUOTE_CHARS


@pytest.mark.parametrize("payload", [{"verdict": "great"}, {}, ["adds_new_work"], "x"])
def test_a_review_without_a_known_verdict_is_refused(payload):
    with pytest.raises(ValueError):
        feedback.parse_review(payload)


def _added(**overrides: Any) -> list[dict[str, Any]]:
    parent, child = _pair_records()
    base = {"candidate": "c1", "parent": "p0", "kind": "parallel", "gate": "rejected",
            "same_seed": True}
    return [{**base, "node_kind": kind,
             **feedback.step_evidence(step, parent_records=parent, child_records=child)}
            for step, kind in feedback.added_steps(PARENT_STATE, CHILD_STATE)]


REVIEW = {"verdict": "restates_inputs", "example": "THINK 4", "note": "It repeats the notes."}


def test_each_added_step_is_reviewed_once_shown_and_kept_for_a_resume(tmp_path):
    client = _Answering({"proposals": []}, REVIEW)
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    ctx = ProposalContext(mode_name="m", added_steps=_added())
    optimizer.propose(ctx, 1, [])
    assert len(client.reviews) == 1, "one call for the step with text output; none for code"
    review = client.reviews[0]
    assert "# Reviewing an added step: `think`" in review
    assert "GIVEN think t4" in review and "THINK 4" in review and "Reasoning 4, reconsidered" in review
    assert "does it contain facts, checks, computations or alternatives its inputs lacked" in review
    section = _section(client.proposals[0], "## Did the added steps contribute?")
    assert ("- `think` (agent, added by c1, parallel; gate rejected): 1 fixed / 1 broken on 6 "
            "shared train item(s); final answer changed on 2/6 (33%; the answer the grader "
            "read); review: restates_inputs "
            '-- "THINK 4" -- It repeats the notes.') in section
    assert "- `check` (code, added by c1" in section and "its records hold no visit of it" in section
    trace = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert [t["verdict"] for t in trace if t["kind"] == "step_review"] == ["restates_inputs"]
    optimizer.propose(ctx, 1, [])
    assert len(client.reviews) == 1, "a retry pays for no review again"
    memory = _Answering({"proposals": []}, REVIEW)
    unstored = AutoresearchOptimizer(memory)  # no run directory: the memory cache alone
    unstored.propose(ctx, 1, [])
    unstored.propose(ctx, 1, [])
    assert len(memory.reviews) == 1
    again = _Answering({"proposals": []}, REVIEW)
    resumed = AutoresearchOptimizer(again)
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    resumed.propose(ctx, 1, [])
    assert again.reviews == [] and "review: restates_inputs" in again.proposals[0]


@pytest.mark.parametrize("review", [RuntimeError("503 overloaded"), {"verdict": "great"},
                                    "not json at all"])
def test_a_failed_review_is_recorded_shown_and_never_fatal(tmp_path, review):
    client = _Answering({"proposals": [{"edit": {"op": "set_knob", "name": "k", "value": 2}}]},
                        review)
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    proposals = optimizer.propose(ProposalContext(added_steps=_added()), 1, [])
    assert len(proposals) == 1, "the proposal is still made"
    section = _section(client.proposals[0], "## Did the added steps contribute?")
    assert "- `think` (agent" in section and "review: not available (" in section
    trace = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    failed = [t for t in trace if t["kind"] == "step_review"]
    assert len(failed) == 1 and failed[0]["error"]
    assert not list((tmp_path / "step_reviews").glob("*.json")), "a failure is not cached"


def _loop_with_an_added_step(tmp_path: Path, monkeypatch: Any, *, gated: bool = False,
                             **overrides: Any):
    """A finished run, then one more child of the seed, proposed by the optimizer in a
    later round, that adds an agent step (rejected by the gate when ``gated``); its
    records and its parent's are train records."""
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    proposer = Recording([[(5, 7)], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": 2, "val_gate": gated, **overrides})
        loop.optimizer = proposer
        loop.run()
    seed = loop.population[0]
    state = json.loads(json.dumps(seed.state))
    state["graph"]["nodes"]["think"] = {"id": "think", "kind": "agent"}
    edit = {"op": "insert_after", "target": "work", "node": {"id": "think", "kind": "agent"}}
    child = Candidate.from_state(state, parent=seed.id, edits=[edit])
    child.evaluated, child.score = True, 0.1
    loop.population.append(child)
    if gated:
        loop._gate_rejected.append(child.id)
    loop.proposals[child.id] = {"id": child.id, "origin": "optimizer", "epoch": 9,
                                "rationale": "add a second opinion"}
    # An earlier round's child that added a step too: not the last round, not reviewed.
    older_state = json.loads(json.dumps(seed.state))
    older_state["graph"]["nodes"]["old_step"] = {"id": "old_step", "kind": "agent"}
    older = Candidate.from_state(older_state, parent=seed.id, edits=[
        {"op": "insert_after", "target": "work", "node": {"id": "old_step", "kind": "agent"}}])
    older.evaluated, older.score = True, 0.1
    loop.population.insert(len(loop.population) - 1, older)
    loop.proposals[older.id] = {"id": older.id, "origin": "optimizer", "epoch": 8}
    if gated:
        loop._gate_rejected.append(older.id)
    parent_records, child_records = _pair_records()
    loop._train_item_records[seed.id] = parent_records
    loop._train_item_records[child.id] = child_records
    loop._train_item_records[older.id] = child_records
    return loop, child


def test_the_loop_hands_the_last_rounds_added_steps_with_train_evidence(tmp_path, monkeypatch):
    loop, child = _loop_with_an_added_step(tmp_path, monkeypatch)
    ctx = loop._context()
    assert [(e["candidate"], e["step"], e["kind"]) for e in ctx.added_steps] == [
        (child.id, "think", "added agent step")]
    assert ctx.added_steps[0]["fixed"] == 1 and ctx.added_steps[0]["samples"]
    ledger = {e["id"]: e for e in ctx.ledger}
    assert ledger[child.id]["kind"] == "added agent step"
    assert ledger[child.id]["touched"] == ["think", "work"]


def test_without_the_review_switch_no_step_is_handed_over(tmp_path, monkeypatch):
    loop, _child = _loop_with_an_added_step(tmp_path, monkeypatch, step_review=False)
    assert loop._context().added_steps == []


def test_the_new_sections_never_show_validation_records(tmp_path, monkeypatch):
    """Under the gate, with val records stored beside train ones: the ledger and the
    step evidence read the train records only."""
    loop, child = _loop_with_an_added_step(tmp_path, monkeypatch, gated=True)
    for cid in (loop.population[0].id, child.id):
        loop._part_results.setdefault(cid, {})["val"] = [{"records": [
            _rec(item, True, {"think": f"VALTEXT {item}", "decide": "v"}) for item in VAL_IDS]}]
    ctx = loop._context()
    text = "\n".join(feedback.step_review_lines(ctx.added_steps)
                     + feedback.ledger_lines(ctx.ledger))
    text += feedback.review_prompt(ctx.added_steps[0])
    assert ctx.added_steps[0]["gate"] == "rejected" and "| rejected |" in text
    assert ctx.added_steps[0]["ran"] == 6, "train items only"
    assert all(s["item"].startswith("t") for s in ctx.added_steps[0]["samples"])
    assert "VALTEXT" not in text
    _no_val(text, loop)


# ── readers' success patterns ───────────────────────────────────────────────


def test_success_patterns_have_their_own_heading_and_room():
    """A reader that found many long failure patterns and two success patterns: both
    success patterns are shown under their own heading, after the failures."""
    from refactory_dsh.optim.base import READER_PATTERNS, _trajectories

    records = [_rec(f"t{i}", i % 2 == 0, {"a": "x" * 3000}) for i in range(40)]
    findings = [{"group": 1, "items": [r["id"] for r in records],
                 "failure_patterns": [{"pattern": f"FAIL-{k} " + "f" * 390, "items": ["t1"],
                                       "evidence": ["q" * 240] * 3} for k in range(20)],
                 "success_patterns": [{"pattern": f"PASS-{k}", "items": ["t0"],
                                       "evidence": ["good"]} for k in range(2)]}]
    text = "\n".join(_trajectories(records, budget=6000, findings=findings))
    heading = "### What the passing items do right (the readers' success patterns)"
    assert heading in text
    after = text[text.index(heading):]
    assert "PASS-0" in after and "PASS-1" in after
    assert text.index("FAIL-0") < text.index(heading)
    assert text.count("FAIL-") <= READER_PATTERNS - 2


# ── structure guidance ──────────────────────────────────────────────────────


def _composed(**fields: Any) -> str:
    ctx = ProposalContext(
        catalogue=[{"op": "insert_after", "kind": "topology", "macro": True,
                    "fields": ["target", "node"], "description": "d"}],
        node_states=[{"id": "a", "kind": "agent", "writes": ["a.txt"]},
                     {"id": "b", "kind": "agent", "writes": ["b.txt"]}],
        edges=[{"source": "a", "target": "b"}], **fields)
    return _section(ctx.render(), "## How nodes compose")


def test_structure_guidance_is_off_by_default_and_said_when_on():
    from refactory_dsh.loop import LoopConfig

    assert LoopConfig().structure_guidance is False
    assert "What makes a new step pay off" not in _composed()
    assert feedback.STRUCTURE_GUIDANCE in _composed(structure_guidance=True)


def test_the_switches_and_their_defaults():
    from refactory_dsh import profiles
    from refactory_dsh.loop import LoopConfig

    config = LoopConfig()
    assert (config.run_ledger, config.lessons, config.step_review,
            config.structure_guidance) == (True, True, True, False)
    settings = profiles.normalize({"run_ledger": 0, "structure_guidance": 1})
    assert settings["run_ledger"] is False and settings["structure_guidance"] is True
    with pytest.raises(profiles.ProfileError):
        profiles.normalize({"lessons": "yes"})


# ── review fixes: same-round comparisons, caps, gate, budget, data flow ─────


def _rescored(tmp_path: Path, monkeypatch: Any):
    """Gated: the child (round 9) was kept as the edited candidate and re-scored in round
    10, and its parent was scored in rounds 9 and 10; the older child (round 8) has no
    parent records from its round."""
    loop, child = _loop_with_an_added_step(tmp_path, monkeypatch, gated=True)
    seed = loop.population[0]
    older = next(c for c in loop.population
                 if (loop.proposals.get(c.id) or {}).get("epoch") == 8)
    parent9, child9 = _pair_records()
    parent10 = [_rec(f"t{i}", True, {"a": "n", "decide": "x"}, "A") for i in range(6)]
    child10 = [_rec(f"t{i}", False, {"a": "n", "think": "t", "decide": "y"}, "Q") for i in range(6)]
    loop._records_by_epoch[seed.id] = {9: parent9, 10: parent10}
    loop._train_item_records[seed.id], loop._records_epoch[seed.id] = parent10, 10
    loop._records_by_epoch[child.id] = {9: child9, 10: child10}
    loop._train_item_records[child.id], loop._records_epoch[child.id] = child10, 10
    loop._records_by_epoch[older.id] = {8: child9}
    loop._records_epoch[older.id] = 8
    return loop, child, older


def test_a_rescored_child_is_compared_in_the_round_it_was_evaluated(tmp_path, monkeypatch):
    loop, child, older = _rescored(tmp_path, monkeypatch)
    ctx = loop._context()
    ledger = {e["id"]: e for e in ctx.ledger}
    assert (ledger[child.id]["fixed"], ledger[child.id]["broken"]) == (1, 1), "round 9 vs 9"
    step = ctx.added_steps[0]
    assert (step["fixed"], step["broken"], step["final_changed"]) == (1, 1, 2)
    assert step["same_seed"] is True
    entries = {e["id"]: e for e in loop._reasoning_entries(loop.leaderboard())}
    assert (entries[child.id]["fixed"], entries[child.id]["broken"]) == (1, 1)
    # The older child's parent has no round-8 records: no counts, said so.
    assert ledger[older.id]["fixed"] is None and ledger[older.id]["same_seed"] is False
    section = "\n".join(feedback.ledger_lines(ctx.ledger))
    row = next(line for line in section.splitlines() if line.startswith(f"| {older.id} |"))
    assert "| n/a (different seed) |" in row
    assert "added agent step: tried 2, kept 0, best train fixed-broken +0" in section


def test_without_the_parents_round_an_added_step_shows_no_counts(tmp_path, monkeypatch):
    loop, child, _older = _rescored(tmp_path, monkeypatch)
    del loop._records_by_epoch[loop.population[0].id][9]
    step = loop._context().added_steps[0]
    assert step["same_seed"] is False and step["shared"] == 0 and step["compared"] == 0
    assert step["samples"], "the review still reads the child's own records"
    line = feedback.step_review_lines([step])[-1]
    assert "fixed/broken and answer changes n/a (different seed" in line
    assert "fixed / " not in line and "final answer changed" not in line


def test_a_candidate_the_gate_has_not_judged_is_left_out(tmp_path, monkeypatch):
    loop, child = _loop_with_an_added_step(tmp_path, monkeypatch, gated=True)
    loop._gate_rejected.remove(child.id)
    ctx = loop._context()
    assert child.id not in {e["id"] for e in ctx.ledger}
    assert child.id not in {e["candidate"] for e in ctx.added_steps}


def _many_steps(count: int) -> list[dict[str, Any]]:
    parent, child = _pair_records()
    base = feedback.step_evidence("think", parent_records=parent, child_records=child)
    return [{**base, "step": f"s{k}", "candidate": "c1", "parent": "p0", "kind": "parallel",
             "node_kind": "agent", "gate": "rejected", "same_seed": True,
             "samples": [{**sample, "output": f"OUT {k}"} for sample in base["samples"]]}
            for k in range(count)]


def test_at_most_the_capped_number_of_steps_are_reviewed_per_round():
    client = _Answering({"proposals": []}, REVIEW)
    AutoresearchOptimizer(client).propose(ProposalContext(added_steps=_many_steps(11)), 1, [])
    assert len(client.reviews) == feedback.STEP_REVIEW_STEPS
    section = _section(client.proposals[0], "## Did the added steps contribute?")
    assert section.count("more added steps than are reviewed per round") == 11 - feedback.STEP_REVIEW_STEPS


def test_a_spent_budget_during_a_review_stops_the_proposal():
    from refactory_dsh.budgets import BudgetExhausted

    client = _Answering({"proposals": []}, BudgetExhausted("token budget spent"))
    with pytest.raises(BudgetExhausted):
        AutoresearchOptimizer(client).propose(ProposalContext(added_steps=_added()), 1, [])
    assert client.proposals == []


def test_the_added_steps_section_stays_within_its_characters():
    entries = [{**entry, "review": {"verdict": "restates_inputs", "example": "e" * 240,
                                    "note": "n" * 300}} for entry in _many_steps(40)]
    lines = feedback.step_review_lines(entries)
    full = [line for line in lines if '-- "eee' in line]
    short = [line for line in lines if line.startswith("- `") and '-- "eee' not in line]
    assert full and short and len(full) + len(short) == 40
    assert sum(len(line) + 1 for line in lines[: lines.index(full[-1]) + 1]) <= feedback.STEP_REVIEW_CHARS
    assert short[-1] == ("- `s39` (added by c1): 1 fixed / 1 broken; final answer changed 2/6; "
                         "review: restates_inputs")
    assert "the remaining steps in one line each" in "\n".join(lines)


FORKED = {"graph": {"nodes": {
    "a": {"id": "a", "kind": "agent", "writes": ["a.txt"]},
    "side_a": {"id": "side_a", "kind": "agent", "reads": ["a.txt"], "writes": ["sa.txt"]},
    "side_b": {"id": "side_b", "kind": "agent", "reads": ["a.txt"], "produces": ["sb"]},
    "join": {"id": "join", "kind": "op", "reads": ["sa.txt"], "produces": ["joined"]},
    "decide": {"id": "decide", "kind": "agent", "prompt_template": "{{ joined }} and {{sb}}"}}}}


def test_the_next_step_is_the_one_that_reads_the_steps_output():
    """In a fork, the sibling branch that ran after ``side_a`` does not read it; the
    step that does (through an op) is ``decide``."""
    records = [_rec("t0", True, {"a": "A", "side_a": "SA", "side_b": "SB", "decide": "D"})]
    shown = feedback.step_evidence("side_a", parent_records=[], child_records=records,
                                   graph=FORKED)["samples"][0]
    assert (shown["next_step"], shown["next_output"]) == ("decide", "D")
    assert feedback.downstream(FORKED, "side_a") == {"join", "decide"}
    assert feedback.downstream(FORKED, "side_b") == {"decide"}
    last = feedback.step_evidence("decide", parent_records=[], child_records=records,
                                  graph=FORKED)["samples"][0]
    assert last["next_step"] == "" and "No recorded step after it reads" in \
        feedback.review_prompt({"step": "decide", "samples": [last]})


def test_a_fresh_run_clears_the_cached_reviews_and_a_resume_keeps_them(tmp_path, monkeypatch):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    with _patch_mode(LevelProbe):
        first = _loop(tmp_path, epochs=1, val_gate=True)
        first.optimizer = Recording([[]])
        first.run()
        cached = first.root / "step_reviews" / "x.json"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text("{}")
        resumed = _loop(tmp_path, epochs=2, val_gate=True, resume=True)
        resumed.optimizer = Recording([[]])
        resumed.run()
        assert resumed.root == first.root and cached.exists(), "a resume keeps them"
        from refactory_dsh.budgets import LEDGER

        (first.root / LEDGER).unlink()  # a fresh run is refused where a budget ledger is
        fresh = _loop(tmp_path, epochs=1, val_gate=True)
        fresh.optimizer = Recording([[]])
        fresh.run()
        assert fresh.root == first.root and not cached.exists()


def test_the_live_sequence_pairs_every_row_on_one_seed(tmp_path, monkeypatch):
    """Arm B's sequence with fresh evidence on and the whole train part each epoch: the
    seed 5 (scored in epoch 1) gets children 7 and 6 in epoch 2; 7 is kept, 6 rejected.
    7's children 3 and 2 are evaluated in epoch 3 and both rejected.

    Each round scores its parent beside its children on the round's sample: the seed in
    epoch 2 (a round that keeps a child), 7 in epoch 3. So every non-seed row has
    numeric counts on same-seed records; the round that keeps nothing is not re-scored
    again after gating, and the evaluations are the candidates plus one per round."""
    from refactory_dsh.loop import Loop
    from test_gate_aware_context import Reasoning

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    _passes_by_level(monkeypatch)
    seeds: list[tuple[int, int, int | None]] = []
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if part == "train":
            seeds.append((_level(candidate), self.epoch, kwargs.get("seed")))
        return original(self, candidate, workdir, part, **kwargs)

    monkeypatch.setattr(Loop, "_measure", measure)
    from refactory_dsh.outerloop import Run

    batches: list[list[tuple[int, Any]]] = []
    many = Run.evaluate_many

    def evaluate_many(self, requests, **kwargs):  # type: ignore[no-untyped-def]
        if all(r.part == "train" for r in requests):
            batches.append([(_level(r.candidate), r.rescore_epoch) for r in requests])
        return many(self, requests, **kwargs)

    monkeypatch.setattr(Run, "evaluate_many", evaluate_many)
    proposer = Reasoning([[(5, 7), (5, 6)], [(7, 3), (7, 2)], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=3, val_gate=True, batch_size=2,
                     max_concurrent_evaluations=3)
        loop.optimizer = proposer
        loop.run()
    ids = {_level(c): c.id for c in loop.population}
    assert loop.config.train_minibatch == 0
    # Two children and three slots: the re-score runs in the children's wave.
    assert sorted(batches[0]) == [(5, 2), (6, None), (7, None)]
    assert sorted(batches[1]) == [(2, None), (3, None), (7, 3)]
    assert sorted(loop._records_by_epoch[ids[5]]) == [1, 2]
    assert sorted(loop._records_by_epoch[ids[7]]) == [2, 3]
    by = {(level, epoch): seed for level, epoch, seed in seeds}
    assert by[(5, 2)] == by[(7, 2)] == by[(6, 2)] and by[(7, 3)] == by[(3, 3)] == by[(2, 3)]
    assert len(seeds) == len(set(seeds)), "nothing scored twice in one epoch"
    rescores = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    rescores = [(r["id"], r["epoch"]) for r in rescores if r.get("event") == "rescore"]
    assert rescores == [(ids[5], 2), (ids[7], 3)], "one per round, none after gating"
    assert loop.evaluations == sum(1 for c in loop.population if c.evaluated) + 2
    rows = {e["id"]: e for e in proposer.contexts[-1].ledger}
    # 5 passes t1-t3, 7 t1-t5, 6 t1-t4, 3 t1, 2 none.
    expected = {7: (2, 0), 6: (1, 0), 3: (0, 4), 2: (0, 5)}
    assert {level: (rows[ids[level]]["fixed"], rows[ids[level]]["broken"])
            for level in expected} == expected
    assert all(rows[ids[level]]["same_seed"] for level in expected)
    table = feedback.ledger_lines(proposer.contexts[-1].ledger)[-1]
    assert table.count("| prompt/setting |") == 4 and "n/a" not in table


def test_the_parents_rescore_takes_one_of_the_concurrent_slots(tmp_path, monkeypatch):
    """With two concurrent evaluations, the two children fill both slots and the
    parent's re-score runs after them: never more than two train measurements in
    flight."""
    import threading
    import time

    from refactory_dsh.loop import Loop
    from test_gate_aware_context import Reasoning

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    lock, state = threading.Lock(), {"now": 0, "most": 0}
    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        if part != "train":
            return original(self, candidate, workdir, part, **kwargs)
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        try:
            time.sleep(0.2)
            return original(self, candidate, workdir, part, **kwargs)
        finally:
            with lock:
                state["now"] -= 1

    monkeypatch.setattr(Loop, "_measure", measure)
    proposer = Reasoning([[(5, 7), (5, 6)], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=True, batch_size=2,
                     max_concurrent_evaluations=2)
        loop.optimizer = proposer
        loop.run()
    assert '"rescore"' in (loop.root / "journal.jsonl").read_text()
    assert state["most"] == 2


def _rescore_run(tmp_path: Path, monkeypatch: Any, plan: list, **overrides: Any):
    from refactory_dsh.loop import Loop
    from test_gate_aware_context import Reasoning

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    proposer = Reasoning(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": len(plan), "val_gate": True, "batch_size": 2,
                                  **overrides})
        loop.optimizer = proposer
        loop.run()
    events = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    return loop, [e for e in events if e.get("event") == "rescore"]


class ParallelProbe(LevelProbe):
    """LevelProbe with its items run 4 at a time (a ``parallelism_knob``), so the
    run-wide item gate is what limits items in flight, as LiveMath's
    ``data_parallelism`` does."""

    name = "level_probe"

    @classmethod
    def declare(cls):  # type: ignore[no-untyped-def]
        from refactory_dsh import nn

        return {**LevelProbe.declare(),
                "par": nn.Knob(default=4, kind="int", lo=1, hi=8, frozen=True,
                               description="items at once")}

    def build(self):  # type: ignore[no-untyped-def]
        import dataclasses

        graph = super().build()
        data = graph.nodes["data"]
        graph.nodes["data"] = dataclasses.replace(data, parallelism_knob="par",
                                                  knobs=(*data.knobs, "par"))
        return graph


@pytest.mark.parametrize("children,slots", [(2, 3), (4, 4)])
def test_the_parents_rescore_stays_under_the_run_wide_item_cap(tmp_path, monkeypatch,
                                                                 children, slots):
    """The parent's re-score beside two children (a slot free) or after four (every
    slot taken), items 4 at a time each, under a run-wide cap of 2 items in flight:
    counted from each item's pass start to its completion, never more than 2 -- the
    re-score's items included, alone or not."""
    import threading

    from refactory_dsh import items as items_module
    from refactory_dsh.loop import Loop
    from test_gate_aware_context import Reasoning

    from refactory_dsh.outerloop import Run

    lock, state = threading.Lock(), {"now": 0, "most": 0, "batch": False}
    start, done = items_module.prepare_pass_dir, items_module.mark_pass_complete
    many = Run.evaluate_many

    def evaluate_many(self, requests, **kwargs):  # type: ignore[no-untyped-def]
        # Items in flight are counted inside batches: a lone evaluation (the seed's)
        # runs at its own parallelism, outside the run-wide gate, as on main; a lone
        # re-score is a batch.
        state["batch"] = len(requests) > 1 or any(r.rescore_epoch is not None for r in requests)
        try:
            return many(self, requests, **kwargs)
        finally:
            state["batch"] = False

    def prepare(*args, **kwargs):  # type: ignore[no-untyped-def]
        with lock:
            state["now"] += 1
            if state["batch"]:
                state["most"] = max(state["most"], state["now"])
        return start(*args, **kwargs)

    def complete(*args, **kwargs):  # type: ignore[no-untyped-def]
        with lock:
            state["now"] -= 1
        return done(*args, **kwargs)

    monkeypatch.setattr(items_module, "prepare_pass_dir", prepare)
    monkeypatch.setattr(items_module, "mark_pass_complete", complete)
    monkeypatch.setattr(Run, "evaluate_many", evaluate_many)
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    plan = [[(5, 7), (5, 6)] if children == 2 else [(5, 1), (5, 2), (5, 3), (5, 4)], []]
    proposer = Reasoning(plan)
    with _patch_mode(ParallelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=True, batch_size=children,
                     max_concurrent_evaluations=slots,
                     max_items_in_flight=2)
        loop.optimizer = proposer
        loop.run()
    assert '"rescore"' in (loop.root / "journal.jsonl").read_text(), "setup: re-scored"
    assert state["most"] == 2, state


@pytest.mark.parametrize("cap", [2, 3, 4, 5, 6])
def test_the_rescore_never_passes_the_evaluation_budget(tmp_path, monkeypatch, cap):
    """The seed (1) and its two children (2) fit a cap of 3; the re-score only goes in
    when the cap holds the children and it, so the run never passes the cap and the
    children are never crowded out by it."""
    loop, rescores = _rescore_run(tmp_path, monkeypatch, [[(5, 7), (5, 6)], [(7, 8), (7, 4)], []],
                                  max_evaluations=cap, max_concurrent_evaluations=3)
    assert loop.evaluations <= cap
    children = [c for c in loop.population if c.parent]
    if cap >= 3:
        assert sum(1 for c in children if c.evaluated and _level(c) in (7, 6)) == 2
    if cap == 3:
        assert rescores == [], "no room beside the children"
    if cap == 4:
        assert [r["epoch"] for r in rescores] == [2]


@pytest.mark.parametrize("cap", [14, 24, 32, 36, 40, 44, 52])
def test_the_rescore_only_starts_where_the_rollout_budget_holds_it(tmp_path, monkeypatch, cap):
    """8 train and 4 val items each; before epoch 2 the seed's train (8) and val (4)
    rollouts are spent. The re-score goes in only when the rollout cap holds both
    children, train and val, and it (12 + 2 x 12 + 8 = 44), so it never crowds out a
    child's scoring or judging, and the run never spends past the cap or past what it
    spends without the re-score; without room it spends exactly the same."""
    from refactory_dsh.loop import Loop

    loop, rescores = _rescore_run(tmp_path / "with", monkeypatch, [[(5, 7), (5, 6)], []],
                                  max_rollouts=cap, max_concurrent_evaluations=3,
                                  max_evaluations=60)
    monkeypatch.setattr(Loop, "_round_parent", lambda self, batch, epoch: None)
    baseline, none = _rescore_run(tmp_path / "without", monkeypatch, [[(5, 7), (5, 6)], []],
                                  max_rollouts=cap, max_concurrent_evaluations=3,
                                  max_evaluations=60)
    assert none == []
    assert bool(rescores) == (12 + 2 * 12 + 8 <= cap), (cap, rescores)
    if rescores:
        assert loop.rollouts <= max(cap, baseline.rollouts)
    else:
        assert loop.rollouts == baseline.rollouts
    assert loop._gate_rejected == baseline._gate_rejected, "the same children judged"


def test_children_that_fill_every_slot_go_first_and_the_rescore_after(tmp_path, monkeypatch):
    """Four children and four slots: the children run in one wave, as without the
    re-score, and the parent's re-score runs after them, alone."""
    from refactory_dsh.outerloop import Run

    batches: list[list[tuple[int, Any]]] = []
    many = Run.evaluate_many

    def evaluate_many(self, requests, **kwargs):  # type: ignore[no-untyped-def]
        if all(r.part == "train" for r in requests):
            batches.append([(_level(r.candidate), r.rescore_epoch) for r in requests])
        return many(self, requests, **kwargs)

    monkeypatch.setattr(Run, "evaluate_many", evaluate_many)
    loop, rescores = _rescore_run(tmp_path, monkeypatch, [[(5, 1), (5, 2), (5, 3), (5, 4)], []],
                                  batch_size=4, max_concurrent_evaluations=4)
    assert [sorted(b) for b in batches[:2]] == [
        [(1, None), (2, None), (3, None), (4, None)], [(5, 2)]]
    assert [r["epoch"] for r in rescores] == [2]


@pytest.mark.parametrize("cap,rescored", [(5, False), (6, True)])
def test_a_rescore_after_full_slots_still_needs_room(tmp_path, monkeypatch, cap, rescored):
    """Four children fill four slots; the seed (1) and they (4) take a cap of 5, so the
    re-score after them is skipped; at 6 it runs. The cap is never passed."""
    loop, rescores = _rescore_run(tmp_path, monkeypatch, [[(5, 1), (5, 2), (5, 3), (5, 4)], []],
                                  batch_size=4, max_concurrent_evaluations=4, max_evaluations=cap)
    assert loop.evaluations <= cap
    assert sum(1 for c in loop.population if c.parent and c.evaluated) == 4
    assert bool(rescores) == rescored


# ── a broken answer is never read in part ───────────────────────────────────


FIRST_EDIT = {"op": "set_knob", "name": "k", "value": 2}
#: The shape of the D-s42 epoch-5 answer: two proposals, the first one's ``edits`` list
#: never closed, so the outer object is not JSON; the first edit alone parses.
BROKEN = ('{"diagnosis": "D-TEXT", "considered": "C-TEXT", "lessons": "L-NEW",\n'
          ' "proposals": [\n'
          '  {"edits": [' + json.dumps(FIRST_EDIT) + ',\n'
          '    {"op": "set_knob", "name": "k2", "patch": [{"kind": "append", "content": "x"}]\n'
          '  }, "rationale": "prompt", "confidence": 0.4},\n'
          '  {"edits": [{"op": "insert_after", "target": "a", "node": {"id": "chk", "kind": "op",'
          ' "call": "refactory_dsh.snippet:run", "code": "def main(i): ..."}}],'
          ' "rationale": "structure", "confidence": 0.3}\n'
          ' ]}')


class _Raw(_Answering):
    """Answers proposal requests with ``text`` as written."""

    def __init__(self, text: str) -> None:
        super().__init__({})
        self.text = text

    def complete(self, prompt: str, *, system: str = "") -> str:
        self.proposals.append(prompt)
        return self.text


@pytest.mark.parametrize("text,why", [
    (BROKEN, "Expecting ',' delimiter at line 5 column 17, here: …"),
    (BROKEN[:200], "its object opened at character 1 is never closed (cut off?"),
])
def test_a_broken_answer_is_read_as_nothing_and_each_asked_proposal_counted(tmp_path, text, why):
    from refactory_dsh.optim.base import proposal_refusal
    from refactory_dsh.optim.random_optimizer import answer_json, extract_json

    assert extract_json(BROKEN) == FIRST_EDIT, "the repro: one edit was read as the answer"
    assert answer_json(BROKEN)[0] == {}
    optimizer = AutoresearchOptimizer(_Raw(text))
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.last_diagnosis = optimizer.last_considered = "AN EARLIER ROUND'S"
    proposals = optimizer.propose(ProposalContext(ask_lessons=True), 2, [])
    assert len(proposals) == 2, "one refused stand-in per proposal asked for"
    for proposal in proposals:
        refused = proposal_refusal(proposal)
        assert refused[0].startswith("E-proposal-shape: your whole answer was not valid JSON ("
                                     + why), refused
        assert "none of its proposals, nor its diagnosis or lessons" in refused[0]
    assert optimizer.last_reply_problem.startswith(why)
    assert (optimizer.last_diagnosis, optimizer.last_considered, optimizer.last_lessons) == (
        "", "", "")
    record = json.loads(optimizer.trace_path.read_text().splitlines()[-1])
    assert (record["proposals"], record["unreadable"]) == (0, 2)
    assert record["error"].startswith("malformed answer: " + why)


def test_an_answer_after_prose_braces_is_still_read():
    """Braces in prose before the answer are not the answer, and not a broken one."""
    from refactory_dsh.optim.random_optimizer import answer_json

    assert answer_json('I weigh {a, b} first. {"proposals": []}') == ({"proposals": []}, "")
    assert answer_json('```json\n{"proposals": [{"edit": {"op": "x"}}]}\n```') == (
        {"proposals": [{"edit": {"op": "x"}}]}, "")
    with pytest.raises(ValueError):
        answer_json("no braces at all")


def test_lessons_not_updated_are_marked_and_cleared_by_new_ones(tmp_path, monkeypatch):
    """An answer that could not be read keeps the lessons, says they were not updated
    then and why; a later answer with lessons clears the mark."""
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    proposer = _BreaksAt([[(5, 7)], [(7, 8)], [(8, 9)], []])
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=3, val_gate=True)
        loop.optimizer = proposer
        loop.run()
    third = _section(proposer.contexts[2].render(), "## Your lessons so far")
    assert "NOT UPDATED in epoch" in third and "BROKEN-WHY" in third and "LESSON-ONE" in third
    assert "these are the earlier ones, unchanged" in third
    assert loop._lessons["text"] == "LESSON-THREE" and "not_updated" not in loop._lessons
    events = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    marked = [e for e in events if e.get("event") == "optimizer_lessons" and e.get("not_updated")]
    assert len(marked) == 1 and marked[0]["text"] == "LESSON-ONE"
    assert marked[0]["not_updated"]["why"] == "BROKEN-WHY"


class _BreaksAt(Lessons):
    """Lessons in rounds 1 and 3; the answer of round ``BREAK`` could not be read."""

    WRITES = {1: "LESSON-ONE", 3: "LESSON-THREE"}
    BREAK = 2

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        out = super().propose(ctx, n, parents)
        self.last_reply_problem = "BROKEN-WHY" if len(self.contexts) == self.BREAK else ""
        return out


def test_a_lessons_mark_survives_a_resume(tmp_path, monkeypatch):
    class Last(_BreaksAt):
        BREAK = 3
        WRITES = {1: "LESSON-ONE"}

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=3, val_gate=True)
        loop.optimizer = Last([[(5, 7)], [(7, 8)], [(8, 9)]])
        loop.run()
    assert loop._lessons["not_updated"]["why"] == "BROKEN-WHY"
    again = Recording([[]])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=4, val_gate=True, resume=True)
        resumed.optimizer = again
        resumed.run()
    shown = _section(again.contexts[0].render(), "## Your lessons so far")
    assert "NOT UPDATED in epoch" in shown and "BROKEN-WHY" in shown and "LESSON-ONE" in shown


def test_lessons_not_updated_without_earlier_lessons_still_say_so():
    lines = feedback.lessons_lines({"not_updated": {"epoch": 1, "generation": 0, "why": "W"}})
    assert lines[0] == "## Your lessons so far" and "NOT UPDATED in epoch 1" in lines[1]
    assert "you have none kept yet" in lines[1] and len(lines) == 2


# ── code steps and gates: measured, not reviewed ────────────────────────────


GATED_CHILD = {"graph": {"nodes": {**CHILD_STATE["graph"]["nodes"],
                                   "g": {"id": "g", "kind": "gate"}}}}


def _visited_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The pair's records, the child's with the visits of its code step ``check`` (output
    on t1 and t4 only) and its gate ``g`` (sends t4 back once, passes the rest)."""
    parent, child = _pair_records()
    for record in child:
        item = record["id"]
        said = item in ("t1", "t4")
        record["steps"] = [
            {"node": "check", "kind": "op", "output": f"signal={item}" if said else "signal=''",
             "digest": item if said else "none", "empty": not said},
            {"node": "g", "kind": "gate", "verdict": "reloop" if item == "t4" else "proceed",
             "reason": "look again" if item == "t4" else ""},
            *([{"node": "g", "kind": "gate", "verdict": "proceed", "reason": ""}]
              if item == "t4" else [])]
    return parent, child


def test_a_code_step_and_a_gate_show_on_which_items_they_acted_and_how_the_answer_moved():
    """Child vs parent: t4 fixed and its answer changed, t0 broken, t5's answer changed.
    The code step gave output on t1 and t4; the gate sent t4 back."""
    parent, child = _visited_records()
    code = feedback.step_evidence("check", parent_records=parent, child_records=child,
                                  graph=GATED_CHILD)
    assert code["samples"] == [] and "decision" not in code
    assert code["visits"] == {
        "kind": "op", "ran": 6, "distinct_outputs": 2, "example": "signal=t1",
        "fired": {"items": ["t1", "t4"], "shared": 2, "changed": 1, "fixed": 1, "broken": 0},
        "silent": {"items": ["t0", "t2", "t3", "t5"], "shared": 4, "changed": 1, "fixed": 0,
                   "broken": 1}}
    gate = feedback.step_evidence("g", parent_records=parent, child_records=child,
                                  graph=GATED_CHILD)["visits"]
    assert gate["routes"] == {
        "proceed": {"items": ["t0", "t1", "t2", "t3", "t5"], "shared": 5, "changed": 1,
                    "fixed": 0, "broken": 1},
        "reloop": {"items": ["t4"], "shared": 1, "changed": 1, "fixed": 1, "broken": 0}}
    assert gate["relooped"]["items"] == ["t4"]
    assert feedback.visit_text(code["visits"]) == (
        "gave output on 2 of 6 item(s) it ran on: t1, t4 (final answer changed on 1/2; "
        '1 fixed / 0 broken), 2 distinct output(s), e.g. "signal=t1"; empty output on 4 '
        "(final answer changed on 1/4; 0 fixed / 1 broken)")
    assert feedback.visit_text(gate) == (
        "ran on 6 item(s); first verdict `proceed` on 5: t0, t1, t2, t3, t5 (final answer "
        "changed on 1/5; 0 fixed / 1 broken); `reloop` on 1: t4 (final answer changed on "
        "1/1; 1 fixed / 0 broken); relooped (visited more than once) on 1: t4 (final answer "
        "changed on 1/1; 1 fixed / 0 broken)")
    assert "final answer changed" not in feedback.visit_text(gate, counts=False)


def test_code_steps_and_gates_are_shown_measured_and_cost_no_review_call():
    parent, child = _visited_records()
    base = {"candidate": "c1", "parent": "p0", "kind": "added code step", "same_seed": True}
    entries = [{**base, "node_kind": kind, **feedback.step_evidence(
        step, parent_records=parent, child_records=child, graph=GATED_CHILD)}
        for step, kind in (("think", "agent"), ("check", "code"), ("g", "gate"))]
    client = _Answering({"proposals": []}, REVIEW)
    AutoresearchOptimizer(client).propose(ProposalContext(added_steps=entries), 1, [])
    assert len(client.reviews) == 1 and "`think`" in client.reviews[0]
    section = _section(client.proposals[0], "## Did the added steps contribute?")
    code = next(line for line in section.splitlines() if line.startswith("- `check`"))
    assert "gave output on 2 of 6 item(s) it ran on: t1, t4" in code
    assert code.endswith("no reviewer (a code step or gate: the measured behaviour above)")
    gate = next(line for line in section.splitlines() if line.startswith("- `g`"))
    assert "first verdict `proceed` on 5" in gate and "relooped (visited more than once) on 1: t4" in gate
    assert "A code step or gate gets no reviewer" in section


# ── a decision step is judged by whether it follows its inputs ──────────────


DECIDING = {"graph": {"nodes": {
    "notes": {"id": "notes", "kind": "agent", "writes": ["notes.txt"]},
    "decide": {"id": "decide", "kind": "agent", "reads": ["notes.txt"], "writes": ["answer.txt"],
               "system_prompt_template": "TEMPLATE",
               "field_bindings": [["system_prompt_template", "decide__system_prompt"]]},
    "extract": {"id": "extract", "kind": "op", "reads": ["answer.txt"], "produces": ["label"]},
    "check": {"id": "check", "kind": "gate", "reads": ["answer.txt"]}}},
    "knobs": {"decide__system_prompt": {"value": "DECIDE-INSTRUCTIONS: pick one."}}}


def _deciding_entry() -> dict[str, Any]:
    records = [_rec(f"t{i}", True, {"notes": f"NOTES {i} recommend B", "decide": "<answer>B</answer>"},
                    "B") for i in range(3)]
    entry = feedback.step_evidence("decide", parent_records=[], child_records=records,
                                   graph=DECIDING)
    return {**entry, "candidate": "c1", "kind": "split", "node_kind": "agent", "same_seed": True}


def test_a_step_whose_output_only_code_or_gates_read_is_a_decision_step():
    assert feedback.decision_step(DECIDING, "decide") is True
    assert feedback.decision_step(DECIDING, "notes") is False, "a model step reads it"
    assert feedback.decision_step(DECIDING, "decide", [{"next_step": "notes"}]) is False
    assert feedback.decision_step(None, "decide") is False
    entry = _deciding_entry()
    assert entry["decision"] is True
    assert entry["instructions"] == "DECIDE-INSTRUCTIONS: pick one.", "the bound knob's value"
    notes = feedback.step_evidence("notes", parent_records=[], child_records=[
        _rec("t0", True, {"notes": "N", "decide": "D"})], graph=DECIDING)
    assert "decision" not in notes


DECISION_REPLY = {"items": [
    {"item": "t0", "inputs_recommend": "B", "decided": "B", "same": True},
    {"item": "t1", "inputs_recommend": "B", "decided": "C", "same": False},
    {"item": "t2", "inputs_recommend": "none", "decided": "B", "same": False}],
    "example": "I choose C", "note": "It overrode its notes once."}


def test_a_decision_step_is_asked_whether_it_decides_as_its_inputs_recommended(tmp_path):
    entry = _deciding_entry()
    prompt = feedback.review_prompt(entry, "m")
    assert prompt.startswith("# Reviewing an added decision step: `decide`")
    assert "repeating the chosen answer is its job" in prompt
    assert "DECIDE-INSTRUCTIONS: pick one." in prompt and "GIVEN decide t0" in prompt
    assert '"inputs_recommend"' in prompt and "restates_inputs" not in prompt
    client = _Answering({"proposals": []}, DECISION_REPLY)
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.propose(ProposalContext(added_steps=[entry]), 1, [])
    section = _section(client.proposals[0], "## Did the added steps contribute?")
    assert ("review: overrides_inputs (decides as its inputs recommended on 1 of 2 item(s) "
            'where they recommended one) -- "I choose C" -- It overrode its notes once.') in section
    again = _Answering({"proposals": []}, DECISION_REPLY)
    resumed = AutoresearchOptimizer(again)
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    resumed.propose(ProposalContext(added_steps=[entry]), 1, [])
    assert again.reviews == [] and "decides as its inputs recommended on 1 of 2" in again.proposals[0]


@pytest.mark.parametrize("rows,verdict,agree,of", [
    ([{"inputs_recommend": "A", "same": True}], "follows_inputs", 1, 1),
    ([{"inputs_recommend": "A", "same": True}, {"inputs_recommend": "A", "same": "yes"}],
     "overrides_inputs", 1, 2),
    ([{"inputs_recommend": "None", "same": True}, {"inputs_recommend": ""}], "no_recommendation", 0, 0),
])
def test_a_decision_review_counts_agreement_where_the_inputs_recommended(rows, verdict, agree, of):
    found = feedback.parse_decision_review({"items": rows})
    assert (found["verdict"], found["agree"], found["of"], found["items"]) == (
        verdict, agree, of, len(rows))
    assert feedback.parse_review(found) == found, "kept and read back for a resume"


@pytest.mark.parametrize("payload", [{}, {"items": []}, {"verdict": "restates_inputs"}, "x"])
def test_a_decision_review_without_items_is_refused(payload):
    with pytest.raises(ValueError):
        feedback.parse_decision_review(payload)


# ── a reviewer reads the step's texts whole ─────────────────────────────────


def test_a_review_reads_each_text_whole_and_any_cut_the_room_forces_is_said():
    long_input = "HEAD " + "x" * 20000 + " TAIL"
    records = [_rec("t0", True, {"a": "A", "think": "y" * 9000, "decide": "z" * 5000})]
    records[0]["exchanges"][1]["prompt"] = long_input
    entry = feedback.step_evidence("think", parent_records=[], child_records=records)
    assert entry["samples"][0]["output"] == "y" * 9000
    prompt = feedback.review_prompt({**entry, "candidate": "c", "kind": "k"})
    assert long_input in prompt and "y" * 9000 in prompt and "z" * 5000 in prompt
    assert "omitted" not in prompt
    fitted = feedback.fit_samples([{"input": long_input, "output": "o" * 3000,
                                    "next_output": "n" * 100}], room=8000)
    assert fitted[0]["next_output"] == "n" * 100 and fitted[0]["output"] == "o" * 3000
    assert fitted[0]["input"].startswith("HEAD ") and fitted[0]["input"].endswith(" TAIL")
    assert "characters omitted" in fitted[0]["input"]
