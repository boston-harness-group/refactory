"""Fixes from the audit of the 2026-11-10 LiveMath full run (outer loop, both runtimes).

- A round's siblings all reach "What the last edit changed", whatever ``batch_size`` says.
- A reloop to a step that cannot see what its gate read (not in its reads, not through a
  fact its prompt reads) hands it those files, the same on the walker and the spine; a
  wired loop's prompts are byte-identical to before; a reloop to an op whose review
  reaches no re-run step is warned of.
- A new or edited step whose output nothing reads is warned of.
- A round's candidates are gated together against the current candidate as the round
  starts: the highest val wins, ties go to the higher train score, then the earlier.
- The sandbox's start-up check has its own limit and says the sandbox was unavailable.
- "structural" is the candidate's own edit, not its lineage's.
- Reader item ids outside the reader's own group are dropped and said.
- Request labels: raw scoring at parsimony 0, the mean named for what it is over, and
  frozen knobs left out of the knob attribution.

Each test fails when the behaviour it names is reverted.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.loop import Loop
from refactory_dsh.optim.base import Candidate, ProposalContext, _edit_change_lines
from refactory_dsh.outerloop import OuterLoopError, RoundStopped, ValidationGate

from test_loop_exhaustion import needs_spine
from test_skillopt import _patch_mode
from test_val_gate import LevelProbe, ScriptedLevels, _gated, _level, _loop
from test_walker_run_record import FakeModel

TESTS = Path(__file__).resolve().parent


# ── a round's siblings all reach "What the last edit changed" ───────────────


def test_every_sibling_of_the_last_round_is_compared_with_its_parent(tmp_path, monkeypatch):
    """Seed 5 gets two children in one round (4 and 3, both rejected, so 5 stays the
    edited candidate). With batch_size 1 -- a portfolio round makes one child per edit
    level whatever the batch size -- both children are still in the comparison."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    loop, _proposer, _report = _gated(tmp_path, monkeypatch, [[(5, 4), (5, 3)], []],
                                      epochs=2, batch_size=2)
    by_level = {_level(c): c.id for c in loop.population}
    assert loop._gate().current_id == by_level[5], "setup: the seed stays current"
    loop.config = dataclasses.replace(loop.config, batch_size=1)
    pairs = loop._edit_pairs(loop.leaderboard())
    assert [p["child"] for p in pairs] == [by_level[4], by_level[3]]
    assert {p["parent"] for p in pairs} == {by_level[5]}


def test_siblings_share_the_comparisons_budget_in_turn():
    """Two children with three changed items each and room for two trajectories: one of
    each child is shown in full, not two of the first; the rest are listed by child."""
    def record(item: str, passed: bool) -> dict[str, Any]:
        return {"id": item, "passed": passed, "score": float(passed),
                "exchanges": [{"node": "s", "prompt": "p", "response": "z" * 2000}]}

    def pair(child: str) -> dict[str, Any]:
        return {"child": child, "parent": "p", "edit": [],
                "parent_records": [record(f"i{k}", True) for k in range(3)],
                "child_records": [record(f"i{k}", False) for k in range(3)]}

    text = "\n".join(_edit_change_lines([pair("A"), pair("B")], budget=11000))
    assert "### Child A of p" in text and "### Child B of p" in text
    assert "#### Item i0 -- broken (child A)" in text
    assert "#### Item i0 -- broken (child B)" in text
    assert "| B | i1 | broken |" in text and "| A | i1 | broken |" in text


# ── a reloop the redone step cannot see, on both runtimes ───────────────────


def _review_loop(wired: bool | str) -> nn.Graph:
    """write -> review -> check (a verdict gate on review.txt) -> done; check sends write
    back on ``reloop``.

    - ``False``, blind: write is shown nothing of the review (B-s44's decide/meta_review).
    - ``True``: write reads review.txt.
    - ``"fact"``: the review's answer is a fact write's prompt renders when set, as design
      mode's steps render their gate's feedback.
    - ``"brief"``: the gate also reads brief.json, which an op writes and whose fact
      ``brief`` write's prompt renders; the review is still not shown.
    - ``"answer"``: the gate also reads write's own answer.txt, whose fact ``answer``
      write's prompt tests; the review is still not shown.
    """
    from refactory_dsh import snippet

    graph = nn.Graph()
    prompt = ("Write it." + ("{% if review_note %}\nLast review: {{review_note}}{% endif %}"
                             if wired == "fact" else "")
              + ("\nBrief: {{brief}}" if wired == "brief" else "")
              + ("{% if answer %}{% endif %}" if wired == "answer" else ""))
    graph.add(nn.Node("write", "agent", prompt_template=prompt, writes=("answer.txt",),
                      reads=(("review.txt",) if wired is True else ()),
                      produces=(("answer",) if wired == "answer" else ())))
    graph.add(nn.Node("review", "agent", prompt_template="Review it.", reads=("answer.txt",),
                      writes=("review.txt",), produces=(("review_note",) if wired == "fact" else ())))
    extra = {"brief": ("brief.json",), "answer": ("answer.txt",)}.get(str(wired), ())
    graph.add(nn.Node("check", "gate", evaluator=snippet.VERDICT, reads=("review.txt", *extra),
                      required_reads=("review.txt",), max_iterations=2))
    graph.add(nn.Node("done", "op", call="loop_ops:noop", reads=("answer.txt",)))
    first = {True: ("start", "loop_ops:no_review_yet", ("review.txt",), ()),
             "brief": ("start", "loop_ops:write_brief", ("brief.json",), ("brief",))}.get(wired)
    if first is not None:  # what write reads or renders exists before the first review
        graph.add(nn.Node(first[0], "op", call=first[1], writes=first[2], produces=first[3]))
        graph.connect("start", "write")
        graph.start = "start"
    graph.connect("write", "review")
    graph.connect("review", "check")
    graph.connect("check", "write", condition="reloop")
    graph.connect("check", "done", condition="proceed")
    return graph


#: The reviewer's answers in turn: a bare ``reloop`` (no reason), then ``proceed``.
REVIEWS = ["reloop", "proceed"]


class Reviewer(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.turn = 0

    def complete(self, prompt: str, *, system: str = "") -> str:
        super().complete(prompt, system=system)
        if "Review it." in prompt:
            self.turn += 1
            return REVIEWS[(self.turn - 1) % len(REVIEWS)]
        return "B"


#: Where the delivery's run-specific preamble ends and the node's own prompt begins.
_TASK_AFTER = "<!-- end refactory worker execution -->\n\n"


def _task(prompt: str) -> str:
    """A worker prompt without its delivery preamble (paths that differ run to run)."""
    return prompt.split(_TASK_AFTER, 1)[-1]


def _walked_writes(graph: nn.Graph, run: Path) -> list[str]:
    model = Reviewer()
    result = nn.execute(graph, run, model=model)
    assert result.exit_ok, result.error
    return [_task(prompt) for _system, prompt in model.prompts if "Review it." not in prompt]


def test_which_reloops_are_blind():
    from refactory_dsh.nn.loops import blind_reloops

    assert blind_reloops(_review_loop(False)) == {"check": {"reloop": ["review.txt"]}}
    assert blind_reloops(_review_loop(True)) == {}, "it reads the review"
    assert blind_reloops(_review_loop("fact")) == {}, "its prompt renders the review's fact"
    # Judged file by file: a fact from another of the gate's files shows not the review.
    assert blind_reloops(_review_loop("brief")) == {"check": {"reloop": ["review.txt"]}}
    assert blind_reloops(_review_loop("answer")) == {"check": {"reloop": ["review.txt"]}}
    keys = _review_loop(False)
    keys.nodes["write"] = dataclasses.replace(keys.nodes["write"], feedback_keys=("note",))
    assert blind_reloops(keys) == {}, "its facts are re-derived from the run's files"
    # A file no step writes (the run's own input): read by the step, it is seen.
    given = _review_loop(False)
    given.nodes["check"] = dataclasses.replace(given.nodes["check"], reads=("review.txt", "brief.md"))
    assert blind_reloops(given) == {"check": {"reloop": ["review.txt", "brief.md"]}}
    given.nodes["write"] = dataclasses.replace(given.nodes["write"], reads=("brief.md",))
    assert blind_reloops(given) == {"check": {"reloop": ["review.txt"]}}
    given.nodes["check"] = dataclasses.replace(given.nodes["check"], reads=("brief.md",),
                                               required_reads=("brief.md",))
    assert blind_reloops(given) == {}, "it reads every file the gate reads"
    person = _review_loop(False)
    person.nodes["check"] = dataclasses.replace(person.nodes["check"], gate_kind="user",
                                                evaluator="")
    assert blind_reloops(person) == {}, "a person's note is the feedback"


def test_the_walker_hands_a_blind_redo_what_the_gate_read(tmp_path):
    writes = _walked_writes(_review_loop(False), tmp_path)
    assert len(writes) == 2 and "Why this step is being redone" not in writes[0]
    assert writes[1].startswith("## Feedback for this visit\n\n## Why this step is being redone\n"
                                "Gate `check` sent this step back (`reloop`) without giving a "
                                "reason.\nIt read these files, which this step is not given:"
                                "\n\n`review.txt`:\nreloop\n\nWrite it.")


def test_a_wired_redo_is_byte_identical_to_before(tmp_path, monkeypatch):
    """A loop whose redone step already sees the review gets exactly the prompts it got
    before this change (the hand-over switched off)."""
    import refactory_dsh.run as runner

    for wired in (True, "fact"):
        now = _walked_writes(_review_loop(wired), tmp_path / f"now-{wired}")
        with monkeypatch.context() as patch:
            patch.setattr(runner, "with_unseen_reads", lambda verdict, *_a, **_k: verdict)
            before = _walked_writes(_review_loop(wired), tmp_path / f"before-{wired}")
        assert now == before and len(now) == 2, wired
        assert "Why this step is being redone" not in now[1]
    assert "Last review: reloop" in now[1]


def test_a_reason_that_already_says_the_file_adds_nothing(tmp_path):
    from refactory_dsh.run import with_unseen_reads

    (tmp_path / "review.txt").write_text("reloop\n\nOption C was dismissed without proof.")
    given = nn.Verdict("reloop", reason="Option C was dismissed without proof.")
    assert with_unseen_reads(given, "g", {"reloop": ["review.txt"]}, tmp_path) is given
    assert with_unseen_reads(given, "g", {"other": ["review.txt"]}, tmp_path) is given
    assert with_unseen_reads(nn.Verdict("proceed"), "g", {"reloop": ["review.txt"]},
                             tmp_path) == "proceed"
    (tmp_path.parent / "secret.txt").write_text("SECRET")
    out = with_unseen_reads("reloop", "g", {"reloop": ["../secret.txt"]}, tmp_path)
    assert "SECRET" not in out.reason and "without giving a reason" in out.reason
    # A short reason contained in a longer review does not hide the review.
    (tmp_path / "long.txt").write_text("reloop\n\n- tests failed\n- the lint step failed too")
    terse = with_unseen_reads(nn.Verdict("reloop", reason="tests failed"), "g",
                              {"reloop": ["long.txt"]}, tmp_path)
    assert "`long.txt`:\nreloop\n\n- tests failed\n- the lint step failed too" in terse.reason
    short = nn.Verdict("reloop", reason="not one of A-E")
    added = with_unseen_reads(short, "g", {"reloop": ["review.txt"]}, tmp_path)
    assert added.reason.startswith("not one of A-E\n\n## Why this step is being redone\n")
    assert "`review.txt`:\nreloop\n\nOption C" in added.reason


def test_the_spines_runner_hands_over_only_what_lowering_declared_blind(tmp_path):
    """The spine's node runner (``python -m refactory_dsh.run``) prints the gate's reason
    before its verdict; the spine hands those lines to the step a reloop goes back to.
    Blind: they carry the review, the walker's text exactly. Wired: nothing is added."""
    from refactory_dsh import lowering
    from refactory_dsh.run import NODE_ENV, WORKDIR_ENV, with_unseen_reads
    from refactory_dsh.nn.loops import blind_reloops

    for wired in (False, True):
        run = tmp_path / str(wired)
        run.mkdir()
        graph = _review_loop(wired)
        lowering.write_declarations({"graph": graph.to_state()}, run)
        (run / "review.txt").write_text("reloop\n")
        env = {**os.environ, WORKDIR_ENV: str(run),
               NODE_ENV: json.dumps({"id": "check", "type": "GateNode", "reads": ["review.txt"]}),
               "PYTHONPATH": os.pathsep.join([str(TESTS), os.environ.get("PYTHONPATH", "")])}
        done = subprocess.run([sys.executable, "-m", "refactory_dsh.run", "refactory_dsh.snippet:verdict"],
                              env=env, cwd=run, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        lines = done.stdout.strip().splitlines()
        assert lines[-1] == "reloop"
        reason = "\n".join(lines[:-1]).strip()
        if wired:
            assert reason == ""
        else:
            walker = with_unseen_reads(nn.Verdict("reloop"), "check",
                                       blind_reloops(graph)["check"], run)
            assert reason == walker.reason and "`review.txt`:\nreloop" in reason


def _spine_writes(graph: nn.Graph, run: Path, tmp_path: Path, monkeypatch: Any) -> list[str]:
    """The prompts write's worker is handed on the real spine drain: the prompt the spine
    rendered, with the reloop feedback the live state holds, as the worker sends them."""
    from refactory_dsh.agent_delivery import with_feedback
    from test_livemath_blocks import _spine

    record = tmp_path / f"drain-{run.name}.jsonl"
    monkeypatch.setenv("DRAIN_RECORD", str(record))
    rc = _spine(graph, run, tmp_path, agent_responses={"review": REVIEWS, "write": "B"})
    assert rc == 0
    sent = [json.loads(line) for line in record.read_text().splitlines()]
    # The worker renders the node's prompt with its feedback (`worker_node`), then the
    # delivery adds its preamble, as the walker does.
    return [with_feedback(_task(e["prompt"]), e.get("feedback", ""))
            for e in sent if e.get("node") == "write"]


@needs_spine
def test_both_runtimes_give_the_same_redo_prompts(tmp_path, monkeypatch):
    """[both runtimes] The blind loop's redo carries the review on the walker and on the
    spine, the same text; the wired loop's prompts are the same on both, with nothing
    added; a step shown another of the gate's files (the brief, its own answer) still
    gets the review, and only the review."""
    for wired in (False, True, "brief", "answer"):
        graph = _review_loop(wired)
        walker = _walked_writes(graph, tmp_path / f"walker-{wired}")
        spine = _spine_writes(graph, tmp_path / f"spine-{wired}", tmp_path, monkeypatch)
        assert walker == spine and len(walker) == 2, (wired, walker, spine)
        handed = wired is not True
        assert ("Why this step is being redone" in walker[1]) is handed
        assert ("`review.txt`:\nreloop" in walker[1]) is handed
        assert "`brief.json`" not in walker[1] and "`answer.txt`" not in walker[1]


def test_a_reloop_to_a_step_that_is_not_an_agent_is_warned_of():
    from refactory_dsh.nn.validate import wiring_warnings

    graph = nn.Graph()
    graph.add(nn.Node("build", "op", call="loop_ops:noop", writes=("prompt.txt",)))
    graph.add(nn.Node("decide", "agent", prompt_template="Decide.", reads=("prompt.txt",),
                      writes=("answer.txt",)))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:three_passes", max_iterations=2,
                      reads=("answer.txt",), produces=("passes",)))
    graph.add(nn.Node("done", "op", call="loop_ops:noop", reads=("answer.txt",)))
    graph.connect("build", "decide")
    graph.connect("decide", "gate")
    graph.connect("gate", "build", condition="reloop")
    graph.connect("gate", "done", condition="proceed")
    warnings = wiring_warnings(graph)
    assert len(warnings) == 1 and warnings[0].startswith("W-reloop-target [gate]")
    assert "'build'" in warnings[0]
    text = ProposalContext(wiring_warnings=warnings).render()
    assert "Wiring warnings" in text and "W-reloop-target [gate]" in text
    good = nn.Graph()
    for node in graph.nodes.values():
        good.add(node)
    good.connect("build", "decide")
    good.connect("decide", "gate")
    good.connect("gate", "decide", condition="reloop")
    good.connect("gate", "done", condition="proceed")
    assert wiring_warnings(good) == []
    # An op that iterates on its own output (no agent wrote what the gate reads): no warning.
    counter = nn.Graph()
    counter.add(nn.Node("step", "op", call="loop_ops:noop", writes=("state.json",)))
    counter.add(nn.Node("check", "gate", evaluator="loop_ops:three_passes", max_iterations=2,
                        reads=("state.json",), produces=("passes",)))
    counter.add(nn.Node("done", "op", call="loop_ops:noop", reads=("state.json",)))
    counter.connect("step", "check")
    counter.connect("check", "step", condition="retry")
    counter.connect("check", "done", condition="proceed")
    assert wiring_warnings(counter) == []
    # A gate target takes the reason itself (design mode's send-backs, dm-226).
    to_gate = nn.Graph()
    for node in good.nodes.values():
        to_gate.add(node)
    to_gate.add(nn.Node("pick", "gate", evaluator="loop_ops:three_passes", max_iterations=2,
                        produces=("passes",)))
    to_gate.connect("build", "decide")
    to_gate.connect("decide", "gate")
    to_gate.connect("gate", "pick", condition="reloop")
    to_gate.connect("gate", "done", condition="proceed")
    to_gate.connect("pick", "build", condition="again")
    to_gate.connect("pick", "done", condition="proceed")
    assert not [w for w in wiring_warnings(to_gate) if "'pick'" in w]


def test_an_edit_that_adds_the_warning_is_told_so_in_its_reasoning_entry():
    """The loop names the warnings a candidate's own edit added (its state's, less its
    parent's), and the reasoning section says them, so a rejected edit's wiring is
    explained too."""
    from refactory_dsh.loop import _wiring_warnings
    from refactory_dsh.optim.base import _reasoning_lines

    graph = nn.Graph()
    graph.add(nn.Node("build", "op", call="loop_ops:noop", writes=("prompt.txt",)))
    graph.add(nn.Node("review", "agent", prompt_template="Review.", writes=("review.txt",)))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:three_passes", max_iterations=2,
                      reads=("review.txt",), produces=("passes",)))
    graph.add(nn.Node("done", "op", call="loop_ops:noop", reads=("prompt.txt",)))
    graph.connect("build", "review")
    graph.connect("review", "gate")
    graph.connect("gate", "build", condition="reloop")
    graph.connect("gate", "done", condition="proceed")
    warnings = [w for w in _wiring_warnings({"graph": graph.to_state()})
                if w.startswith("W-reloop-target")]
    assert warnings and warnings[0].startswith("W-reloop-target")
    assert _wiring_warnings({"graph": {"nodes": "not a graph"}}) == []
    text = "\n".join(_reasoning_lines([{"id": "c", "epoch": 1, "gate": "rejected",
                                        "warnings": warnings, "edit": []}]))
    assert f"Wiring warning this edit added: {warnings[0]}" in text


def _prompt_built_by_an_op(with_check: bool) -> nn.Graph:
    """compare -> [meta_check] -> decide_prompt (an op that reads only compare's notes
    and sets the prompt fact) -> decide -> finish: B-s45 3eb923de5c6b's shape."""
    graph = nn.Graph()
    graph.add(nn.Node("compare", "agent", prompt_template="Compare.", writes=("compare_notes.txt",)))
    if with_check:
        graph.add(nn.Node("meta_check", "agent", prompt_template="Check the meta-option.",
                          reads=("compare_notes.txt",), writes=("meta_check.txt",)))
    graph.add(nn.Node("decide_prompt", "op", call="loop_ops:noop", reads=("compare_notes.txt",),
                      produces=("decide_prompt",)))
    graph.add(nn.Node("decide", "agent", prompt_template="{{decide_prompt}}\nWeigh ALL the notes above.",
                      writes=("answer.txt",)))
    graph.add(nn.Node("finish", "op", call="loop_ops:noop", reads=("answer.txt",)))
    order = ["compare", *(["meta_check"] if with_check else []), "decide_prompt", "decide", "finish"]
    for source, target in zip(order, order[1:]):
        graph.connect(source, target)
    return graph


def test_a_new_step_whose_output_nothing_reads_is_warned_of():
    from refactory_dsh.nn.validate import wiring_warnings

    seed, edited = _prompt_built_by_an_op(False), _prompt_built_by_an_op(True)
    warnings = wiring_warnings(edited, seed)
    assert len(warnings) == 1 and warnings[0].startswith("W-unread-output [meta_check]")
    assert "'meta_check.txt'" in warnings[0] and "`reads`" in warnings[0]
    assert wiring_warnings(seed, seed) == [], "the mode's own steps are not judged"
    # Once a later step declares the file, the step is read.
    wired = _prompt_built_by_an_op(True)
    wired.nodes["decide_prompt"] = dataclasses.replace(
        wired.nodes["decide_prompt"], reads=("compare_notes.txt", "meta_check.txt"))
    assert wiring_warnings(wired, seed) == []
    # A step that sets a fact, or the last step (its files are the output), is left alone.
    facts = _prompt_built_by_an_op(True)
    facts.nodes["meta_check"] = dataclasses.replace(facts.nodes["meta_check"], produces=("meta",))
    assert wiring_warnings(facts, seed) == []
    last = _prompt_built_by_an_op(False)
    last.nodes["finish"] = dataclasses.replace(last.nodes["finish"], writes=("result.txt",))
    assert wiring_warnings(last, seed) == [], "the last step's files are the output"


def test_the_loop_says_an_unread_step_beside_the_pipeline_and_in_the_edits_entry():
    from refactory_dsh.loop import _wiring_warnings

    seed = {"graph": _prompt_built_by_an_op(False).to_state()}
    edited = {"graph": _prompt_built_by_an_op(True).to_state()}
    assert [w.split(" ", 2)[:2] for w in _wiring_warnings(edited, seed)] == [
        ["W-unread-output", "[meta_check]:"]]
    assert _wiring_warnings(edited, edited) == []


# ── a round is gated together ───────────────────────────────────────────────


class _NoRollouts:
    def evaluate(self, *_args: Any, **_kwargs: Any) -> None:
        raise OuterLoopError("the rollout budget is spent")


def _gate(current: float, scores: dict[str, float]) -> ValidationGate:
    gate = ValidationGate()
    gate.current_id, gate.current_score = "cur", current
    gate.best_id, gate.best_score = "cur", current
    gate.cache = dict(scores)
    return gate


def _cand(cid: str, train: float | None) -> Candidate:
    return Candidate(id=cid, parent="cur", state={}, score=train)


def test_a_round_is_judged_against_its_starting_candidate_whatever_the_order():
    for order in (("a", "b"), ("b", "a")):
        gate = _gate(0.5, {"a": 0.6, "b": 0.7})
        accepted, rejected = gate.judge_round(_NoRollouts(), [_cand(c, 0.1) for c in order])
        assert [c.id for c in accepted] == ["b"] and [c.id for c in rejected] == ["a"]
        assert (gate.current_id, gate.current_score, gate.best_id) == ("b", 0.7, "b")


def test_a_val_tie_goes_to_the_higher_train_score_then_to_the_earlier():
    gate = _gate(0.5, {"a": 0.7, "b": 0.7})
    accepted, _ = gate.judge_round(_NoRollouts(), [_cand("a", 0.2), _cand("b", 0.3)])
    assert [c.id for c in accepted] == ["b"]
    gate = _gate(0.5, {"a": 0.7, "b": 0.7})
    accepted, _ = gate.judge_round(_NoRollouts(), [_cand("a", None), _cand("b", 0.0)])
    assert [c.id for c in accepted] == ["b"], "a failed train run counts lowest"
    gate = _gate(0.5, {"a": 0.7, "b": 0.7})
    accepted, _ = gate.judge_round(_NoRollouts(), [_cand("a", 0.3), _cand("b", 0.3)])
    assert [c.id for c in accepted] == ["a"]


def test_a_round_that_beats_nothing_keeps_the_current_candidate():
    gate = _gate(0.5, {"a": 0.5, "b": 0.4})
    accepted, rejected = gate.judge_round(_NoRollouts(), [_cand("a", 0.9), _cand("b", 0.9)])
    assert accepted == [] and [c.id for c in rejected] == ["a", "b"]
    assert gate.current_id == "cur"


def test_a_refused_rollout_judges_what_was_scored_and_leaves_the_rest():
    gate = _gate(0.5, {"a": 0.6})
    with pytest.raises(RoundStopped) as stopped:
        gate.judge_round(_NoRollouts(), [_cand("a", 0.1), _cand("b", 0.1), _cand("c", 0.1)])
    assert [c.id for c in stopped.value.accepted] == ["a"] and stopped.value.rejected == []
    assert gate.current_id == "a"


def test_the_loop_keeps_the_rounds_tie_winner_on_train_score(tmp_path, monkeypatch):
    """B-s42 round 3: two siblings tie on val above the current candidate. 7 comes first,
    6 has the higher train score (LevelProbe's train score falls as the level rises):
    6 is kept and becomes current, 7 is rejected."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    val = (lambda run, cand: 0.5 if _level(cand) == 5 else 0.8)
    loop, _proposer, _report = _gated(tmp_path, monkeypatch, [[(5, 7), (5, 6)], []],
                                      val=val, epochs=2, batch_size=2)
    by_id = {c.id: c for c in loop.population}
    assert by_id[[c.id for c in loop.population if _level(c) == 6][0]].score > \
        by_id[[c.id for c in loop.population if _level(c) == 7][0]].score, "setup"
    assert [_level(by_id[cid]) for cid in loop._gate_accepted] == [5, 6]
    assert [_level(by_id[cid]) for cid in loop._gate_rejected] == [7]
    assert _level(by_id[loop._gate().current_id]) == 6


# ── the sandbox's start-up is not the code's time ───────────────────────────


def _slow_start(monkeypatch: Any, delay: float) -> None:
    """Stand the sandbox in with the bare driver behind a ``delay`` of start-up, so the
    two clocks can be told apart on any machine (no namespaces needed)."""
    from refactory_dsh import snippet

    monkeypatch.setattr(snippet, "_command", lambda extra: [
        "bash", "-c", f'sleep {delay}; exec "$0" -I -S -c "$1"', sys.executable, snippet._DRIVER])


def test_a_slow_sandbox_start_is_not_counted_against_the_code(monkeypatch):
    """Two seconds to start and a one-second code limit: the code still runs, because
    the code's clock starts once the sandbox has (``STARTED``)."""
    from refactory_dsh import snippet

    _slow_start(monkeypatch, 2)
    assert snippet._sandboxed("def main(inp):\n    return inp + 1\n", 41, timeout_s=1,
                              extra=(), startup_s=30) == 42


def test_a_start_over_its_allowance_is_the_sandbox_unavailable_not_the_code(monkeypatch):
    from refactory_dsh import snippet

    _slow_start(monkeypatch, 3)
    monkeypatch.setattr(snippet, "STARTUP_S", 0.5)
    monkeypatch.setattr(snippet, "available", lambda: (True, ""))
    with pytest.raises(snippet.SandboxUnavailable,
                       match="cannot run here: the sandbox was unavailable: the sandbox did "
                             "not start within 0.5 s") as raised:
        snippet.execute("def main(inp):\n    return 1\n", {})
    assert "ran over" not in str(raised.value)
    # The code's own overrun is still the code's.
    _slow_start(monkeypatch, 0)
    with pytest.raises(snippet.SnippetTimeout, match="the code ran over its 1 s limit"):
        snippet._sandboxed("def main(inp):\n    while True:\n        pass\n", {},
                           timeout_s=1, extra=(), startup_s=30)


def test_a_failed_check_is_kept_for_a_bounded_time_across_processes(tmp_path, monkeypatch):
    """A busy machine pays one check per PROBE_CACHE_S, not one per op: a failed check is
    read back from the shared file by a fresh process, and taken again once it is old."""
    import time

    from refactory_dsh import snippet

    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path))
    checks: list[int] = []

    def check():  # type: ignore[no-untyped-def]
        checks.append(1)
        return False, "the sandbox was unavailable: it did not start within 60 s"

    monkeypatch.setattr(snippet, "_check_sandbox", check)
    monkeypatch.setattr(snippet, "_PROBED", None)
    assert snippet.available()[0] is False and len(checks) == 1
    monkeypatch.setattr(snippet, "_PROBED", None)  # a fresh process
    assert snippet.available() == (False, "the sandbox was unavailable: it did not start "
                                          "within 60 s")
    assert len(checks) == 1, "read from the shared file, not checked again"
    later = time.time() + snippet.PROBE_CACHE_S + 1
    monkeypatch.setattr(snippet, "_PROBED", None)
    monkeypatch.setattr(time, "time", lambda: later)
    snippet.available()
    assert len(checks) == 2, "an old answer is checked again"


def test_the_offer_and_the_refusal_use_one_answer_per_request(tmp_path, monkeypatch):
    """The request's answer (``ProposalContext.sandbox``) decides both: the loop asks
    once per request, and the refusal judges by that answer, never asking again."""
    from refactory_dsh import snippet
    from refactory_dsh.optim.base import _code_node_refusals

    answers = iter([(True, ""), (False, "busy")])
    asked: list[int] = []

    def flipping():  # type: ignore[no-untyped-def]
        asked.append(1)
        return next(answers)

    monkeypatch.setattr(snippet, "available", flipping)
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, val_gate=True, epochs=1, allow_code_nodes=True)
        loop.init()
        ctx = loop._context()
    assert ctx.sandbox == (True, "") and ctx.code_nodes is True and len(asked) == 1
    proposer = ScriptedLevels([])
    proposer.propose_valid(ctx, 1, parents=[])
    assert proposer.sandbox == (True, "")
    coded = nn.Graph()
    coded.add(nn.Node("c", "op", call=snippet.RUN, code="def main(inp):\n    return {}\n"))
    code_node = {"graph": coded.to_state()}
    monkeypatch.setattr(snippet, "available", lambda: pytest.fail("asked again"))
    assert _code_node_refusals({}, code_node, True, (True, "")) == []
    assert "busy" in _code_node_refusals({}, code_node, True, (False, "busy"))[0]


# ── "structural" is the candidate's own edit ────────────────────────────────


def test_a_prompt_child_of_a_structural_parent_is_not_structural():
    edits = [{"op": "add_node", "node": {}, "position": {"start": True}},
             {"op": "set_knob", "name": "x", "value": 1}]
    child = Candidate(id="c", parent="p", state={}, edits=edits, step_edits=1)
    assert not child.is_structural() and child.describe()["structural"] is False
    assert child.lineage_is_structural()
    parent = Candidate(id="p", parent="s", state={}, edits=edits[:1], step_edits=1)
    assert parent.is_structural()
    compound = Candidate(id="d", parent="s", state={}, edits=edits, step_edits=2)
    assert compound.is_structural()


def test_structural_wins_count_only_a_structural_step(tmp_path, monkeypatch):
    """``_structural_wins`` and ``_structural_stats`` label a candidate by its own edit."""
    edits = [{"op": "add_node", "node": {}, "position": {"start": True}},
             {"op": "set_knob", "name": "x", "value": 1}]
    parent = Candidate(id="p", parent=None, state={}, edits=[], score=0.1, evaluated=True)
    child = Candidate(id="c", parent="p", state={}, edits=edits, step_edits=1, score=0.5,
                      evaluated=True)
    fake = SimpleNamespace(population=[parent, child], rejections=[], _row=lambda c: {"id": c.id},
                           _op_mix=lambda: {})
    assert Loop._structural_wins(fake, [child, parent]) == []
    assert Loop._structural_stats(fake)["structural_candidates"] == 0


# ── reader item ids ─────────────────────────────────────────────────────────


def test_a_readers_ids_outside_its_group_are_dropped_and_said(tmp_path):
    from refactory_dsh.optim.base import _trajectories
    from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer

    class Reader:
        def complete(self, prompt: str, *, system: str = "") -> str:
            return json.dumps({"failure_patterns": [
                {"pattern": "skips option C", "step": "decide", "items": ["a", "v:8"],
                 "evidence": ["q"]}], "success_patterns": []})

    optimizer = AutoresearchOptimizer(Reader(), k=1)
    findings, failures = optimizer._read(ProposalContext(mode_name="m"),
                                         [[("a", "block a")], [("b", "block b")]])
    assert not failures
    first = findings[0]["failure_patterns"][0]
    assert first["items"] == ["a"] and first["not_in_group"] == ["v:8"]
    records = [{"id": i, "passed": False, "score": 0.0, "prediction": "x" * 3000,
                "exchanges": [{"node": "s", "prompt": "p", "response": "r" * 3000}]}
               for i in ("a", "b")]
    text = "\n".join(_trajectories(records, budget=6000, findings=findings))
    assert "items a; it also named v:8, not an item of its group, so dropped)" in text


# ── request labels ──────────────────────────────────────────────────────────


def test_scoring_is_raw_without_a_size_cost():
    raw = Loop.scoring(SimpleNamespace(config=SimpleNamespace(parsimony=0.0)))
    assert raw["compared"] == "raw" and "penalized" not in json.dumps(raw)
    sized = Loop.scoring(SimpleNamespace(config=SimpleNamespace(parsimony=0.01)))
    assert sized["compared"] == "penalized"


def test_the_mean_is_named_for_what_it_is_over(tmp_path, monkeypatch):
    text = ProposalContext(mean_score=0.25, mean_over=7).render()
    assert "Mean over the 7 candidate(s) scored so far: 0.25." in text
    assert "Batch mean" not in text
    seen: list[ProposalContext] = []

    class Recording(ScriptedLevels):
        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            seen.append(ctx)
            return super().propose(ctx, n, parents)

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, val_gate=True, epochs=2)
        loop.optimizer = Recording([[(5, 6)], []])
        loop.run()
    assert seen and all(ctx.mean_over > 0 for ctx in seen)
    assert "Mean over the" in seen[-1].render() and "Batch mean" not in seen[-1].render()


def test_frozen_knobs_are_left_out_of_the_knob_attribution(tmp_path, monkeypatch):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, c: _level(c) / 10))
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, val_gate=True, epochs=2)
        loop.optimizer = ScriptedLevels([[(5, 6)], []])
        loop.run()
    evaluated = loop._evaluated()
    for candidate in evaluated:
        candidate.state["knobs"]["workers"] = {
            "value": 4, "spec": {"kind": "int", "default": 1, "lo": 1, "hi": 64, "frozen": True}}
    attribution = loop._knob_attribution(evaluated)
    assert "level" in attribution and "workers" not in attribution
    loop.config = dataclasses.replace(loop.config, frozen=("level",))
    assert "level" not in loop._knob_attribution(evaluated)
    loop.config = dataclasses.replace(loop.config, frozen=())
    loop.parameters = {**loop.parameters,
                       "level": dataclasses.replace(loop.parameters["level"], frozen=True)}
    assert "level" not in loop._knob_attribution(evaluated)


# ── the portfolio report's shadowed rule ────────────────────────────────────


def test_a_sibling_beaten_by_one_proposed_after_it_is_shadowed(tmp_path, monkeypatch):
    """Seed 5 gets 6 (proposed first) and then 7: both beat 5 on val, 7 wins the round.
    6 is rejected although it beat its parent: shadowed, though the winner came after."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    loop, _proposer, _report = _gated(tmp_path, monkeypatch, [[(5, 6), (5, 7)], []],
                                      epochs=2, batch_size=2)
    by_level = {_level(c): c.id for c in loop.population}
    for level, edit_level in ((6, "prompt_knob"), (7, "pipeline")):
        loop.proposals[by_level[level]]["level"] = edit_level
    report = loop._portfolio_report()
    [round_] = [entry["candidates"] for entry in report["per_round"]
                if {c["id"] for c in entry["candidates"]} == {by_level[6], by_level[7]}]
    first, second = round_
    assert (first["id"], first["gate"], first["shadowed"]) == (by_level[6], "rejected", True)
    assert second["id"] == by_level[7] and second["kept"] and not second["shadowed"]


# ── design mode declares what synthesize_qa reads ───────────────────────────


def test_design_modes_qa_rerun_is_declared_where_it_is_read():
    """``synthesize_qa`` folds in ``reviews/qa-rerun.md`` (``design_ops.synthesize_qa``)
    and declares it, so the wiring check finds a reader for ``qa_director_rerun``."""
    from refactory_dsh.design import DesignMode
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.validate import wiring_warnings

    graph = Graph.from_state(DesignMode().state_dict()["graph"])
    assert "reviews/qa-rerun.md" in graph.nodes["synthesize_qa"].reads
    assert "reviews/qa-rerun.md" in graph.nodes["qa_director_rerun"].writes
    assert not [w for w in wiring_warnings(graph) if w.startswith("W-unread-output")]
