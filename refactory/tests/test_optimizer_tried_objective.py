"""Two follow-ups of a LiveMath smoke test of the outer loop's optimizer requests.

- **The memory pass sees what the epoch tried.** Its request lists every change the
  optimizer proposed in the round the start-of-epoch pipeline opened: each evaluated
  candidate with its kind, its edits in short, its train fixed/broken counts against its
  parent and the gate's word, and each proposal refused before it ran with why -- the
  facts the proposal request already shows, never a validation score; bounded, newest
  first, the cut said. Without it a pass with nothing kept wrote "no changes made",
  though candidates had been tried and refused.
- **The optimizer is told the real objective.** With parsimony 0 every optimizer
  request (proposal, analysts, merges, memory) says the score is the task's own score
  and that inference cost, latency and step count are not part of it unless the task's
  own score measures them, and -- when the score is the items' mean -- the proposal
  request's trace table drops its ``cost`` row (gradient.json keeps it). Under the
  validation gate each also says how the gate decides. With parsimony
  above 0 the size cost is said as it is scored.

Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
from typing import Any

from refactory_dsh.optim import ProposalContext, analysts, memory
from refactory_dsh.optim.base import UNREADABLE_PROPOSAL, Candidate, Proposal, objective_text

from test_analysts import _ctx, _rec as arec
from test_optimizer_memory import Remembering, _events, _rec, _run, _section
from test_skillopt import _patch_mode
from test_val_gate import LevelProbe, _level, _loop


# ── what the epoch tried, in the memory request ─────────────────────────────


class Refusing(Remembering):
    """The scripted proposer; in the rounds ``extra`` names, refused proposals go first
    (an unreadable entry and an op outside the run's permitted edits), so the round
    still accepts its scripted move after them."""

    def __init__(self, plan: list, extra: dict[int, list[Proposal]]) -> None:
        super().__init__(plan)
        self.extra = extra
        self.rounds = 0

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.rounds += 1
        return [*self.extra.get(self.rounds, []), *super().propose(ctx, n, parents)]


def _refused() -> list[Proposal]:
    return [Proposal(edit={UNREADABLE_PROPOSAL: "NO-OP-FIELD"}),
            Proposal(edit={"op": "bypass", "target": "work"}, rationale="R")]


def test_a_pass_with_nothing_kept_lists_what_was_tried_and_refused(tmp_path, monkeypatch):
    """Seed 5, 7 kept in epoch 2; in round 2, 6 is tried (rejected at the gate) after two
    refusals. Epoch 3's pass compares 7 with itself, and its request still says what
    was tried, why the refusals were refused, and no validation score."""
    plan = [[(5, 7)], [(7, 6)], []]
    proposer = Refusing(plan, {2: _refused()})
    loop, _ = _run(tmp_path, monkeypatch, plan, proposer=proposer)
    by_level = {_level(c): c for c in loop.population}
    passes = _events(loop, "optimizer_memory")
    assert [p["same"] for p in passes] == [False, False]
    prompt = proposer.asked[1]["prompt"]
    tried = _section(prompt, "## What was tried this epoch")
    assert "proposed in round 2" in tried
    # Nothing kept: the incumbent 7 is set against the rejected 6 on the item 6 broke.
    assert "Nothing this epoch replaced it; the comparison below sets it against" in prompt
    assert passes[1]["rejected"]["child"] == by_level[6].id
    six = by_level[6].id
    (line,) = [l for l in tried.splitlines() if l.startswith(f"- `{six}`")]
    assert f"from `{by_level[7].id}`" in line and "knob change: prompt/setting" in line
    # Its edits are whole in the changes section; the tried line points there, said once.
    assert 'its edits are under "What the rejected change changed" above' in line
    assert prompt.count('{"op": "set_knob", "name": "level", "value": 6}') == 1
    assert '"value": 6' in _section(prompt, "## What the rejected change changed")
    assert f"train score {by_level[6].score}" in line
    # Its counts and the gate's word are in the per-change tally, said once there.
    (tally,) = [l for l in tried.splitlines() if f"`{six}`" in l and l is not line
                and not l.startswith(f"- `{six}`")]
    # 7 passes t1-t5, 6 passes t1-t4: one broken, none fixed.
    assert "fixed 0 / broken 1 of" in tally and "gate: rejected" in tally
    assert "fixed 0 / broken 1" not in line and "gate:" not in line
    refusals = [l for l in tried.splitlines() if l.startswith("- refused before it ran")]
    assert len(refusals) == 2
    # Newest first: the bypass was refused after the unreadable entry.
    assert "E-policy" in refusals[0] and "structural change" in refusals[0]
    assert '{"op": "bypass", "target": "work"}' in refusals[0]
    assert "E-proposal-shape" in refusals[1]
    assert "attempt 1; no readable edit)" in refusals[1] and "change:" not in refusals[1]
    assert tried.index(f"`{six}`") < tried.index("- refused"), "evaluated ones first"
    # Epoch 2's pass lists round 1 only: 7, kept, and no refusal.
    first = _section(proposer.asked[0]["prompt"], "## What was tried this epoch")
    assert f"`{by_level[7].id}`" in first and f"`{six}`" not in first and "- refused" not in first
    # No validation score anywhere in either request (`_val`: 0.1234 + level / 1000).
    for asked in proposer.asked:
        for level in (5, 6, 7):
            assert f"{0.1234 + level / 1000:g}" not in asked["prompt"]
    assert "val score" not in tried.lower()
    # Structural or knob is the candidate's own edits' (`Candidate.is_structural`).
    monkeypatch.setattr(Candidate, "is_structural", lambda self: True)
    (entry,) = loop._tried_in_round(2)[0]
    assert entry["structural"] is True and entry["id"] == six


def test_a_resumed_run_keeps_each_rounds_refusals(tmp_path, monkeypatch):
    plan = [[(5, 7)], [(7, 6)]]
    proposer = Refusing(plan, {2: _refused()})
    loop, _ = _run(tmp_path, monkeypatch, plan, proposer=proposer)
    assert [r["epoch"] for r in _events(loop, "rejection")] == [2, 2]
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=2, val_gate=True, resume=True)
        resumed._replay_if_asked()
    assert {k: len(v) for k, v in resumed._round_refusals.items() if v} == {2: 2}
    assert resumed._tried_in_round(2) == loop._tried_in_round(2)


def test_a_journal_without_refusal_epochs_files_them_under_the_open_epoch(tmp_path, monkeypatch):
    plan = [[(5, 7)], [(7, 6)]]
    proposer = Refusing(plan, {2: _refused()})
    loop, _ = _run(tmp_path, monkeypatch, plan, proposer=proposer)
    journal = loop.root / "journal.jsonl"
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    for line in lines:
        if line.get("event") == "rejection":
            line.pop("epoch")
    journal.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=2, val_gate=True, resume=True)
        resumed._replay_if_asked()
    assert {k: len(v) for k, v in resumed._round_refusals.items() if v} == {2: 2}


def _tried(n: int, **extra: Any) -> list[dict[str, Any]]:
    return [{"id": f"c{i}", "parent": "p", "round": 4, "kind": "prompt/setting",
             "structural": False, "edits": "set_knob: k" + "x" * 200, "score": 0.5,
             "exit_ok": True, "evaluated": True, "fixed": 1, "broken": 0, "shared": 10,
             "same_seed": True, "gate": "rejected", **extra} for i in range(n)]


def test_the_section_is_bounded_newest_first_with_the_cut_said():
    lines = memory.tried_lines(_tried(60), [{"round": 4, "attempt": 1, "kind": "",
                                             "structural": None, "edits": "(no readable edit)",
                                             "reasons": ["E-noop: X"]}],
                               round_=4, current="c59", gated=True)
    text = "\n".join(lines)
    assert len(text) <= memory.TRIED_CHARS
    assert memory.TRIED_CHARS == 16000
    assert "`c59`" in text and "`c0`" not in text, "the newest kept, the oldest cut"
    assert text.index("`c59`") < text.index("`c58`")
    assert lines[-1].startswith("[cut: ") and lines[-1].endswith(
        " entries not shown: refusals first, then the oldest evaluated]")
    assert "E-noop" not in text, "the refusal, listed last, is cut first"
    cut = int(lines[-1].split()[1])
    assert cut == 61 - sum(1 for line in lines if line.startswith("- "))
    whole = "\n".join(memory.tried_lines(_tried(2), [], round_=4, current="c1", gated=True))
    assert "[cut" not in whole and "`c0`" in whole


def test_the_entries_say_the_gate_the_integrity_verdict_and_the_refusal():
    integrity = ("E-integrity: an integrity review judged what this change adds an exploit "
                 "(dataset_artifact): it keys on the option order")
    lines = memory.tried_lines(
        _tried(1, integrity="suspect: grader_artifact", same_seed=False, fixed=None,
               broken=None),
        [{"round": 4, "attempt": 2, "kind": "added agent step", "structural": True,
          "edits": "insert_after: check, solve", "reasons": [integrity]}],
        round_=4, current="c0", gated=True)
    text = "\n".join(lines)
    assert "gate: rejected" in text and "integrity: suspect: grader_artifact" in text
    assert "its parent was not scored with the same seed" in text
    assert "structural change: added agent step" in text and "attempt 2" in text
    assert "an exploit (dataset_artifact)" in text
    ungated = "\n".join(memory.tried_lines(_tried(1, gate=None), [], round_=4, current="c0",
                                           gated=False))
    assert "gate:" not in ungated and "the pipeline now" in ungated
    empty = memory.tried_lines([], [], round_=4, current="c0", gated=True)
    assert empty[-1] == memory.NOTHING_TRIED == "(nothing was proposed in that round)"


def test_with_nothing_tried_the_unchanged_pipeline_points_nowhere():
    entries, _ = memory.pairs([_rec("a", True)], [_rec("a", False)], ["a"])
    side = {"id": "c1", "pipeline": "P", "rollout": "R"}

    def built(tried: list[str]) -> str:
        return memory.build_prompt(epoch=3, mode_name="m", prev=side, curr=side, changes=[],
                                   previous_note="", entries=entries, tried=tried)[0]

    nothing = built(memory.tried_lines([], [], round_=2, current="c1", gated=True))
    assert "Nothing this epoch replaced it: the comparison" in nothing
    assert memory.NOTHING_TRIED in nothing
    some = built(memory.tried_lines(_tried(1), [], round_=2, current="c1", gated=True))
    assert "Nothing this epoch replaced it (what was tried and not kept is under" in some


def test_edits_say_what_each_change_was_its_text_included():
    # S1-s45: "loop" alone could not tell this judge's criteria from loops in general.
    text = memory.edits_text([{"op": "set_knob", "name": "a", "value": "NEW-TEXT"},
                              {"op": "insert_after", "target": "solve",
                               "node": {"id": "check", "kind": "agent",
                                        "prompt_template": "CHECK-CRITERIA"}}])
    assert "NEW-TEXT" in text and "CHECK-CRITERIA" in text and "solve" in text
    # A knob's change against the parent's value is its diff, not the whole text.
    diff = memory.edits_text([{"op": "set_knob", "name": "a", "value": "one\nTWO\nthree"}],
                             {"a": "one\nthree"})
    assert diff.startswith("set_knob a: 9 -> 13 chars; diff: ") and "+TWO" in diff
    assert "\n" not in text and "\n" not in diff, "one line: nothing in it reads as a heading"
    assert memory.edits_text({UNREADABLE_PROPOSAL: "x"}) == "(no readable edit)"
    assert len(memory.edits_text([{"op": "set_knob", "name": "k" * 9000}])) <= \
        memory.TRIED_EDITS_CHARS


def test_a_clipped_trace_names_the_memory_requests_room():
    prev = [_rec(f"i{n}", False, "x" * 60000) for n in range(4)]
    curr = [_rec(f"i{n}", n % 2 == 0, "y" * 60000) for n in range(4)]
    entries, _ = memory.pairs(prev, curr, [f"i{n}" for n in range(4)])
    text, _clip = memory.build_prompt(
        epoch=3, mode_name="m", prev={"id": "c1", "pipeline": "P", "rollout": "R"},
        curr={"id": "c2", "pipeline": "P", "rollout": "R"}, changes=[], previous_note="",
        entries=entries, room=200000)
    assert "to fit the memory request's room." in text and "analyst's room" not in text
    big = arec("t1", False, steps={"draft": ("z" * 90000, "z" * 90000)})
    assert analysts.fit_item(big, 20000).endswith("to fit the analyst's room.)")


# ── the objective ───────────────────────────────────────────────────────────


UNLESS = "are not part of the score unless the task's own score measures them"
GRADIENT = {"epoch": 1, "scoring": {"compared": "raw"},
            "cost": {"n": 1, "mean_s": 294.75, "inner_model_calls": 40}}


def test_with_no_size_cost_the_request_says_cost_is_not_scored_and_drops_the_cost_row():
    text = ProposalContext(parsimony=0.0, traces=[GRADIENT], score_is_item_mean=True,
                           selection="single").render()
    said = _section(text, "## What the score measures")
    assert ("The score is the task's own score (the mean of the items' scores) and nothing "
            "else.") in said
    assert "inference cost, latency, and the number of steps or model calls" in said
    assert "are not part of the score unless the task's own score measures them" in said
    assert "validation gate" in said and "a tie with the current candidate is rejected" in said
    trace = _section(text, "## Evidence from the traces")
    assert "| cost |" not in trace and "294.75" not in text and "| scoring |" in trace
    assert text.index("## What the score measures") < text.index("## The pipeline you may edit")
    # A score that is not the items' mean may carry cost itself: the row stays.
    other = ProposalContext(parsimony=0.0, traces=[GRADIENT]).render()
    assert "| cost |" in _section(other, "## Evidence from the traces")
    assert "validation gate" not in _section(other, "## What the score measures")


def test_with_a_size_cost_the_request_says_it_as_scored_and_keeps_the_cost_row():
    text = ProposalContext(parsimony=0.01, traces=[GRADIENT], score_is_item_mean=True).render()
    said = _section(text, "## What the score measures")
    assert "minus a size cost of 0.01 per node" in said and "raw score - 0.01 x nodes" in said
    assert "nothing else" not in said and "the number of steps" not in said
    assert "| cost |" in _section(text, "## Evidence from the traces")
    gated = objective_text(0.01, gated=True)
    assert ("it keeps a candidate only when that score beats the current candidate's, and "
            "that score carries no size cost; a train score, size cost included, only breaks "
            "a tie between candidates that beat the current one by the same validation score; "
            "a tie with the current candidate is rejected.") in gated
    assert "Apart from that size cost, inference cost, latency and the number of model " \
           f"calls {UNLESS}." in gated
    plain = objective_text(0.0, gated=True)
    assert ("it keeps a candidate only when that score beats the current candidate's; a "
            "train score only breaks a tie between candidates that beat the current one by "
            "the same validation score") in plain and "size cost" not in plain
    unknown = ProposalContext(traces=[GRADIENT]).render()
    assert "## What the score measures" not in unknown and "| cost |" in unknown


def test_analysts_and_merges_are_told_the_objective():
    ctx = _ctx([arec("t1", False), arec("t2", True)])
    assert "What the score measures" not in analysts.analyst_prompt(
        ctx, "failure", 1, 1, [arec("t1", False)], "PIPE")
    ctx = ProposalContext(**{**ctx.__dict__, "parsimony": 0.0})
    asked = analysts.analyst_prompt(ctx, "failure", 1, 1, [arec("t1", False)], "PIPE")
    merged = analysts.merge_prompt(ctx, "final", [[], []], 0, "PIPE")
    for text in (asked, merged):
        assert "## What the score measures\nThe score is the task's own score" in text
        assert "are not part of the score unless the task's own score measures them" in text


def test_the_memory_request_is_told_the_objective_and_the_run_keeps_the_cost(
        tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []], parsimony=0.0)
    prompt = proposer.asked[0]["prompt"]
    assert "## What the score measures\nThe score is the task's own score" in prompt
    assert "are not part of the score unless the task's own score measures them" in prompt and "a tie with the current candidate is rejected" in prompt
    ctx = proposer.contexts[-1]
    request = ctx.render()
    assert "are not part of the score unless the task's own score measures them" in request
    assert ("| cost |" in request) is not ctx.score_is_item_mean
    gradient = json.loads((loop.root / "gradient.json").read_text())
    assert gradient["cost"]["n"] >= 1, "the run's own record keeps the cost"
    _, proposer2 = _run(tmp_path / "b", monkeypatch, [[(5, 7)], []], parsimony=0.05)
    assert "size cost of 0.05 per node" in proposer2.asked[0]["prompt"]
    assert "| cost |" in proposer2.contexts[-1].render()


def test_no_edit_is_dropped_and_the_cut_is_said():
    """Past the bound every edit keeps its head (op and what it names) and the cut is said
    in edits and characters; successive set_knobs of one knob are diffed against the
    value in between."""
    many = [{"op": "set_knob", "name": f"k{i}", "value": "V" * 400} for i in range(12)]
    text = memory.edits_text(many)
    assert all(f"k{i}" in text for i in range(12)), "no edit vanishes"
    assert "[cut: 12 edits /" in text and "chars]" in text
    wide = {"op": "insert_after", "target": "solve",
            "node": {"id": "check", "kind": "agent", "reads": [f"f{i}.txt" for i in range(400)]}}
    late = memory.edits_text([wide, {"op": "set_knob", "name": "LATE", "value": "x"}])
    assert "LATE" in late and "[cut: 1 edits /" in late
    twice = memory.edits_text([{"op": "set_knob", "name": "a", "value": "one\ntwo"},
                               {"op": "set_knob", "name": "a", "value": "one\ntwo\nthree"}],
                              {"a": "one"})
    assert twice.count("+two") == 1 and "+three" in twice, "the second diffs against the first"


def test_a_cut_edit_keeps_its_wiring_and_cuts_its_text_in_the_middle():
    """Past the bound a cut edit keeps its op, the step it names, where it sends back, what
    it sees and its limit whole; only its long text is cut, head and tail kept; and the
    whole stays within ``TRIED_EDITS_CHARS``."""
    # Texts first and last, the wiring between: a cut of the whole line would lose it.
    loop = {"text": "HEAD-OF-CRITERIA " + "c" * 6000 + " TAIL-OF-CRITERIA", "op": "loop",
            "step": "solve", "back_to": "plan", "judge": "agent", "sees": ["task", "solve"],
            "max_iterations": 2, "system": "SYSTEM-HEAD " + "s" * 3000 + " SYSTEM-TAIL"}
    knobs = [{"op": "set_knob", "name": f"k{i}", "value": "V" * 900} for i in range(3)]
    text = memory.edits_text([loop, *knobs])
    assert len(text) <= memory.TRIED_EDITS_CHARS
    for wiring in ('"step": "solve"', '"back_to": "plan"', '"sees": ["task", "solve"]',
                   '"max_iterations": 2', '"op": "loop"'):
        assert wiring in text, wiring
    assert "HEAD-OF-CRITERIA" in text and "TAIL-OF-CRITERIA" in text and "middle cut" in text
    assert all(f"k{i}" in text for i in range(3)) and "[cut: " in text


def test_a_tiny_room_keeps_the_cut_marker_whole():
    """With room for little more than the cut's marker, the marker is kept whole (and the
    edit's head), the text before it dropped, never the marker sliced."""
    from refactory_dsh.optim.memory import _shrunk_edit

    part = "set_knob k: 10 -> 9000 chars; diff: " + "x" * 9000
    short = _shrunk_edit(None, part, "[set_knob k]", 30)
    assert short.endswith("middle cut]…") and short.startswith("[set_knob k]")
    assert f"[{len(part)} chars, middle cut]" in short
