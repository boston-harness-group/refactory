"""How a structural edit moved the items it changed (`optim.edit_effect`), from the item
records alone: the new steps that ran, their gates' verdicts, the steps that ran again,
and -- where the Task reads a reply as an answer -- whether the first reply was already
right. Shown in the next proposer request and given to the memory call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh.optim.base import _edit_change_lines
from refactory_dsh.optim.edit_effect import edit_effect, effect_lines
from refactory_dsh.optim import memory as memory_mod
from refactory_dsh.optim.memory import tried_lines
from refactory_dsh.task import Task, reply_judge
from refactory_packages.livemath.tasks import LiveMathTask

REAL = json.loads((Path(__file__).parent / "fixtures" / "edit_effect" / "s1_s45_loop.json")
                  .read_text())
#: The steps the S1 loop added: its judge and the gate reading the judge's verdict.
LOOP_STEPS = {"solve_check", "solve_check_judge"}


def _judge():
    return LiveMathTask.__new__(LiveMathTask).reply_correct


def test_the_real_s1_s45_loop_broke_five_items_whose_first_solve_was_right():
    """The audit's finding, computed: of the 8 items the loop broke, the solver's first
    reply was right on 5 before the judge sent it back, and wrong on its last."""
    effect = edit_effect(REAL["parent"], REAL["child"], _judge(), new_steps=LOOP_STEPS)
    broken = effect["broken"]
    assert broken["items"] == 8
    assert broken["new"] == {"solve_check": 8, "solve_check_judge": 8}
    assert broken["said"]["solve_check"]["reloop"] == 8
    assert broken["reran"]["solve"] == 8
    assert broken["first_right"]["solve"] == 5 and broken["then_changed"]["solve"] == 5
    assert broken["never_right"]["solve"] == 3
    text = "\n".join(effect_lines(effect))
    assert "How the edit moved its 8 broken item(s) (parent right, child wrong)" in text
    assert "gate solve_check gave reloop on 8" in text and "at least once" in text
    assert ("solve ran more often than in the parent on 8; of those 8, solve's first reply "
            "was already right on 5 and changed to a wrong one on 5 of those; never right "
            "on 3") in text


def _rec(item, passed, replies, gates=()):
    return {"id": item, "passed": passed, "label": "B",
            "metadata": {"choices": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"}],
                         "correct_choice": {"label": "B", "text": "b"}},
            "exchanges": [{"node": node, "response": text} for node, text in replies],
            "steps": [{"node": gate, "kind": "gate", "verdict": verdict}
                      for gate, verdict in gates]}


def test_without_a_reading_it_says_the_steps_and_verdicts_and_no_first_reply():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [_rec("i1", False, [("solve", "<answer>B</answer>"), ("judge", "reloop"),
                                ("solve", "<answer>A</answer>")], [("check", "reloop")])]
    effect = edit_effect(parent, child, new_steps={"judge", "check"})
    assert effect["broken"]["new"] == {"judge": 1, "check": 1}
    assert effect["broken"]["reran"] == {"solve": 1} and effect["broken"]["first_right"] == {}
    assert "the task reads no reply as an answer" in "\n".join(effect_lines(effect))
    with_reading = edit_effect(parent, child, _judge(), new_steps={"judge", "check"})
    assert with_reading["broken"]["then_changed"] == {"solve": 1}


def test_an_edit_that_added_no_step_says_nothing():
    """A text edit moves items too, but the records cannot say how: no effect, no line."""
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [_rec("i1", False, [("solve", "<answer>A</answer>")])]
    assert edit_effect(parent, child, _judge(), new_steps=set()) == {}
    assert effect_lines({}) == []


def test_the_request_and_memory_say_it_beside_the_items_counts():
    effect = edit_effect(REAL["parent"], REAL["child"], _judge(), new_steps=LOOP_STEPS)
    pair = {"child": "44548d72f753", "parent": "7691424216b6", "edit": [],
            "child_records": REAL["child"], "parent_records": REAL["parent"], "effect": effect}
    request = "\n".join(_edit_change_lines([pair]))
    assert "8 broken (parent right, child wrong)" in request
    assert "How the edit moved its 8 broken item(s)" in request
    entry = {"id": "44548d72f753", "parent": "7691424216b6", "round": 4, "kind": "loop",
             "structural": True, "edits": "loop solve", "score": 0.5, "exit_ok": True,
             "evaluated": True, "fixed": 1, "broken": 8, "shared": 35, "effect": effect}
    memory = "\n".join(tried_lines([entry], [], round_=4, current=None, gated=True))
    assert "first reply was already right on 5" in memory


def test_a_task_without_a_reading_gives_no_judge_and_livemath_reads_its_replies():
    assert Task().reply_correct("anything", {}) is None
    judge = _judge()
    meta = REAL["child"][0]["metadata"]
    right = meta["correct_choice"]["label"]
    assert judge(f"<answer>{right}</answer>", meta) is True
    assert judge("no choice named here at all", {"choices": [], "correct_choice": {}}) is None
    from refactory_packages.livemath.mode import Solver

    found = reply_judge(Solver().state_dict())
    assert found is not None and found(f"<answer>{right}</answer>", meta) is True


def test_the_loop_gives_the_request_and_memory_each_structural_childs_effect(
        tmp_path, monkeypatch):
    """A round's child, its records and its parent's: the request's pair and memory's
    tried entry carry how it moved its items, read with the mode's Task."""
    import dataclasses

    from refactory_dsh.loop import Loop
    from refactory_dsh.optim.base import Candidate
    from test_val_gate import _gated, _level

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    loop, _proposer, _report = _gated(tmp_path, monkeypatch, [[(5, 4)], []], epochs=2,
                                      batch_size=1)
    loop.config = dataclasses.replace(loop.config, val_gate=False)
    by_level = {_level(c): c for c in loop.population}
    child, parent = by_level[4], by_level[5]
    loop._train_item_records[child.id] = REAL["child"]
    loop._train_item_records[parent.id] = REAL["parent"]
    monkeypatch.setattr(Loop, "_reply_judge", lambda self: _judge())
    # The probe's two candidates share a graph; the real loop's child added LOOP_STEPS.
    monkeypatch.setattr("refactory_dsh.loop._new_steps", lambda p, c: set(LOOP_STEPS))
    (pair,) = loop._edit_pairs(loop.leaderboard())
    assert pair["effect"]["broken"]["then_changed"] == {"solve": 5}
    monkeypatch.setattr(Candidate, "is_structural", lambda self: True)
    round_ = (loop.proposals.get(child.id) or {}).get("epoch")
    (entry,) = loop._tried_in_round(round_)[0]
    assert entry["effect"]["broken"]["first_right"] == {"solve": 5}
    # The tally the memory pass was given is kept with the note it wrote, for the proposer.
    loop._pending_tally = ["TALLY-LINE"]
    loop._keep_memory({"epoch": 2}, {"text": "NOTE"})
    assert loop._memory["tally"] == ["TALLY-LINE"]


# ── finding 15: each kind tried, against the noise ──────────────────────────


def _tried(name: str) -> dict:
    """A tried entry for the real loop in fixture ``name``, as the loop makes one."""
    from refactory_dsh.optim.base import edit_change

    fixture = json.loads((Path(__file__).parent / "fixtures" / "edit_effect" / name).read_text())
    change = edit_change(fixture["parent"], fixture["child"])
    return {"id": fixture["child_id"], "parent": fixture["parent_id"], "round": 1,
            "kind": "loop", "structural": True, "edits": "loop solve", "score": 0.5,
            "exit_ok": True, "evaluated": True, "fixed": len(change["fixed"]),
            "broken": len(change["broken"]), "shared": fixture["shared"], "gate": "rejected",
            "effect": edit_effect(fixture["parent"], fixture["child"], _judge(),
                                  new_steps=LOOP_STEPS)}, fixture


@pytest.mark.parametrize("name,fixed,broken", [("s1_s43_loop.json", 2, 9),
                                               ("s1_s45_loop.json", 0, 8)])
def test_memory_gets_each_kind_s_tries_and_the_noise_they_sit_in(name, fixed, broken):
    """The real S1 loops, each tried once: the memory call is told the try's counts and
    the gate's word once (in the tally), the noise -- one of the 17 validation items is
    0.059 -- and how the try moved its items, so its note can say what went wrong instead
    of banning loops."""
    entry, fixture = _tried(name)
    tally = memory_mod.kind_tally_lines([entry], val_items=fixture["val_items"], gated=True)
    text = "\n".join(tried_lines([entry], [], round_=1, current=None, gated=True, tally=tally))
    assert f"one item of the validation part moves its mean by 1/17 = 0.059" in text
    assert (f"- loop: 1 try: `{entry['id']}` fixed {fixed} / broken {broken} of 35 train "
            "items, gate: rejected") in text
    assert text.count(f"fixed {fixed} / broken {broken}") == 1
    assert "first reply was already right" in text
    # The noise rule is the memory call's, once: in its system prompt, not again here.
    assert "never" in memory_mod.MEMORY_SYSTEM and "inside the noise" in memory_mod.MEMORY_SYSTEM
    request = memory_mod.MEMORY_SYSTEM + "\n" + text
    assert request.count("no evidence") == 1, "the memory request repeats the noise rule"


def test_the_proposer_sees_the_same_tally_once_beside_the_note():
    from refactory_dsh.optim.base import ProposalContext

    entry, fixture = _tried("s1_s45_loop.json")
    tally = memory_mod.kind_tally_lines([entry], val_items=17, gated=True)
    text = ProposalContext(memory={"text": "NOTE-TEXT", "epoch": 2, "tally": tally}).render()
    note = text.split("## Optimizer memory", 1)[1]
    assert note.index(memory_mod.TALLY_HEADING) < note.index("NOTE-TEXT")
    assert text.count(memory_mod.TALLY_HEADING) == 1
    assert text.count(f"`{entry['id']}` fixed 0 / broken 8") == 1
    assert text.count(memory_mod.TALLY_NOISE_RULE) == 1


# ── what the line may claim, on any graph ──────────────────────────────────


def _meta_rec(item, passed, replies, gates=(), error=None):
    record = _rec(item, passed, replies, gates)
    if error:
        record.update(error_text=error, exchanges=[])
    return record


def test_new_is_the_child_graph_s_steps_and_a_failed_parent_item_is_left_out():
    """The real fixture's fixed item: its parent timed out with no calls, so solve 'ran
    there now and not before' -- but solve is not new, and nothing ran to compare."""
    fixed = edit_effect(REAL["parent"], REAL["child"], _judge(), new_steps=LOOP_STEPS)
    assert "solve" not in (fixed.get("fixed") or {}).get("new", {})
    parent = [_meta_rec("i1", False, [], error="CompletionTimeout: timed out")]
    child = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    effect = edit_effect(parent, child, _judge(), new_steps={"other"})
    assert effect["fixed"]["parent_failed"] == 1 and effect["fixed"]["new"] == {}
    assert "1 left out: the parent's run of the item failed" in "\n".join(effect_lines(effect))


def test_a_rerun_is_more_calls_than_the_parent_made_on_the_item():
    """A parent that already looped solve, and a child adding an unrelated step: the loop
    is not the edit's. A step the parent called several times has no known first reply."""
    loop = [("solve", "<answer>A</answer>"), ("solve", "<answer>B</answer>")]
    parent = [_rec("i1", True, loop)]
    child = [_rec("i1", False, [*loop, ("note", "x")])]
    effect = edit_effect(parent, child, _judge(), new_steps={"note"})
    assert effect["broken"]["reran"] == {} and effect["broken"]["new"] == {"note": 1}
    more = [_rec("i1", False, [*loop, ("solve", "<answer>A</answer>"), ("note", "x")])]
    effect = edit_effect(parent, more, _judge(), new_steps={"note"})
    assert effect["broken"]["reran"] == {"solve": 1}
    assert effect["broken"]["unread_runs"] == {"solve": 1}
    assert effect["broken"]["first_right"] == {}
    assert "its first reply is not known" in "\n".join(effect_lines(effect))


def test_a_last_reply_the_task_cannot_read_is_not_a_change_to_a_wrong_one():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [_rec("i1", False, [("solve", "<answer>B</answer>"), ("j", "reloop"),
                                ("solve", "no label at all")])]
    effect = edit_effect(parent, child, _judge(), new_steps={"j"})
    assert effect["broken"]["first_right"] == {"solve": 1}
    assert effect["broken"]["then_changed"] == {}


def test_a_reply_right_after_a_wrong_first_is_not_never_right():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [_rec("i1", False, [("solve", "<answer>A</answer>"), ("j", "reloop"),
                                ("solve", "<answer>B</answer>"), ("j", "reloop"),
                                ("solve", "<answer>A</answer>")])]
    effect = edit_effect(parent, child, _judge(), new_steps={"j"})
    assert effect["broken"]["never_right"] == {}


def test_an_unreadable_first_reply_and_a_failing_reading_are_said():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [_rec("i1", False, [("solve", "nothing"), ("j", "r"), ("solve", "<answer>A</answer>")])]
    effect = edit_effect(parent, child, _judge(), new_steps={"j"})
    assert "not readable as an answer on any of those 1" in "\n".join(effect_lines(effect))

    def broken_reading(reply, meta):
        raise ValueError("bad reading")

    effect = edit_effect(parent, child, broken_reading, new_steps={"j"})
    assert "reading of a reply failed 2 time(s), first: ValueError: bad reading" in \
        "\n".join(effect_lines(effect))


def test_a_group_with_nothing_to_say_is_left_out():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")]),
              _rec("i2", False, [("solve", "<answer>A</answer>")])]
    child = [_rec("i1", False, [("solve", "<answer>A</answer>"), ("j", "x")]),
             _rec("i2", True, [("solve", "<answer>B</answer>")])]
    effect = edit_effect(parent, child, _judge(), new_steps={"j"})
    assert set(effect) == {"broken", "reads_replies"}
    assert all(not line.endswith(": .") for line in effect_lines(effect))


def test_an_op_the_edit_added_is_said():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [{**_rec("i1", False, [("solve", "<answer>A</answer>")]),
              "steps": [{"node": "pick", "kind": "op"}]}]
    effect = edit_effect(parent, child, _judge(), new_steps={"pick"})
    assert effect["broken"]["new"] == {"pick": 1}


def test_a_graph_or_task_that_cannot_load_is_loud():
    from refactory_dsh.task import reply_judge

    with pytest.raises(Exception):
        reply_judge({"graph": {"nodes": {"d": {"id": "d", "kind": "data",
                                               "subgraph_entry": "x", "subgraph_exit": "x",
                                               "task_ref": "no.such.module:Task"}},
                               "edges": []}})


def test_only_a_structural_candidate_gets_an_effect(tmp_path, monkeypatch):
    import dataclasses

    from refactory_dsh.loop import Loop
    from refactory_dsh.optim.base import Candidate
    from test_val_gate import _gated, _level

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    loop, _proposer, _report = _gated(tmp_path, monkeypatch, [[(5, 4)], []], epochs=2,
                                      batch_size=1)
    loop.config = dataclasses.replace(loop.config, val_gate=False)
    by_level = {_level(c): c for c in loop.population}
    child, parent = by_level[4], by_level[5]
    loop._train_item_records[child.id] = REAL["child"]
    loop._train_item_records[parent.id] = REAL["parent"]
    monkeypatch.setattr(Loop, "_reply_judge", lambda self: _judge())
    monkeypatch.setattr("refactory_dsh.loop._new_steps", lambda p, c: set(LOOP_STEPS))
    monkeypatch.setattr(Candidate, "is_structural", lambda self: False)
    round_ = (loop.proposals.get(child.id) or {}).get("epoch")
    (entry,) = loop._tried_in_round(round_)[0]
    assert "effect" not in entry


def test_a_fixed_item_whose_parent_failed_is_counted_apart():
    parent = [_meta_rec("i1", False, [], error="CompletionTimeout: timed out"),
              _rec("i2", False, [("solve", "<answer>A</answer>")])]
    child = [_rec("i1", True, [("solve", "<answer>B</answer>"), ("j", "ok")]),
             _rec("i2", True, [("solve", "<answer>B</answer>"), ("j", "ok")])]
    group = edit_effect(parent, child, _judge(), new_steps={"j"})["fixed"]
    assert (group["items"], group["parent_failed"], group["new"]) == (2, 1, {"j": 1})
    assert "1 left out: the parent's run of the item failed" in "\n".join(
        effect_lines({"fixed": group, "reads_replies": True}))


def test_a_rerun_whose_first_replies_are_all_unreadable_says_only_that():
    parent = [_rec("i1", True, [("solve", "<answer>B</answer>")])]
    child = [_rec("i1", False, [("solve", "nothing"), ("j", "r"), ("solve", "still nothing")])]
    text = "\n".join(effect_lines(edit_effect(parent, child, _judge(), new_steps={"j"})))
    assert "solve's first reply was not readable as an answer on any of those 1" in text
    assert "right on 0" not in text
