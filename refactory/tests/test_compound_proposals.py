"""Compound proposals: a proposal may carry an ordered list of edits (``edits``),
applied together to the parent as one candidate and judged once.

Each edit goes through the per-edit checks in order, the combined state through every
state rule, and a refusal names the edit it is about. ``max_edits_per_proposal`` caps
the list (default 4; 1 is single edits). A proposal with one ``edit`` works as before.
Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn, profiles
from refactory_dsh.loop import Loop, LoopError
from refactory_dsh.optim import AutoresearchOptimizer, Candidate, Optimizer, Proposal, ProposalContext
from refactory_dsh.optim.base import DEFAULT_MAX_EDITS
from refactory_dsh.outerloop import Run

from test_skillopt import TASK_REF, _patch_mode
from test_train_minibatch import _events
from test_val_gate import _loop

import structure_fuzz as sf


class TwoLevels(nn.Mode):
    """The toy Task's data node and two integer knobs; train score falls with ``level``."""

    name = "two_levels"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"level": nn.Knob(default=5, kind="int", lo=0, hi=9, description="a level"),
                "extra": nn.Knob(default=5, kind="int", lo=0, hi=9, description="another"),
                "note": nn.Knob(default="Be careful.", kind="str", description="a text")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work",
            task_ref=TASK_REF, writes=("current_item.json",), produces=("item_scores",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item", reads=("current_item.json",),
            writes=("seen.json",), produces=("seen",), knobs=("level", "extra", "note"),
        ))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        return 1.0 - int(self.values()["level"]) / 10


def _knob(name: str, value: Any) -> dict[str, Any]:
    return {"op": "set_knob", "name": name, "value": value}


def _seed() -> Candidate:
    return Candidate.from_state(TwoLevels().state_dict())


def _attempt(proposal: Proposal, parent: Candidate | None = None, *, seen=(), max_edits=None):  # type: ignore[no-untyped-def]
    optimizer = Optimizer()
    if max_edits is not None:
        optimizer.max_edits = max_edits
    parent = parent or _seed()
    return optimizer.attempt(proposal, parent, TwoLevels().invariants(), True, (),
                             [parent.id, *seen])


def _values(candidate: Candidate) -> dict[str, Any]:
    return {k: v["value"] for k, v in candidate.state["knobs"].items()}


# ── the proposal and the attempt path ──────────────────────────────────────


def test_a_list_is_applied_in_order_as_one_candidate():
    seed = _seed()
    append = {"op": "set_knob", "name": "note", "patch": [{"kind": "append", "content": "Twice."}]}
    edits = [_knob("level", 3), _knob("note", "Check."), append]
    child = _attempt(Proposal(edits=tuple(edits), parent=seed.id), seed)
    assert isinstance(child, Candidate), child
    values = _values(child)
    assert values["level"] == 3 and values["note"].startswith("Check.") and "Twice." in values["note"], \
        "in order: the patch applies to the value the edit before it set"
    assert child.parent == seed.id and child.edits == edits and child.own_edits() == edits
    assert child.step_edits == 3 and len(child.edit_reports) == 3
    grandchild = _attempt(Proposal(edit=_knob("extra", 1)), child)
    assert grandchild.own_edits() == [_knob("extra", 1)], "a single edit's own edits: one"
    assert grandchild.edits == [*edits, _knob("extra", 1)]


def test_a_refusal_names_the_edit_that_failed():
    refused = _attempt(Proposal(edits=(_knob("level", 3), _knob("ghost", 1))))
    assert isinstance(refused, list) and "[edit 2 of 2: set_knob]" in refused[0], refused
    assert refused[0].split(" ", 1)[0].rstrip(":").startswith("E-"), "the code stays first"
    unknown = _attempt(Proposal(edits=(_knob("level", 3), {"op": "teleport"})))
    assert "[edit 2 of 2: teleport]" in unknown[0], unknown
    not_object = _attempt(Proposal(edits=("level=3", _knob("level", 3))))
    assert not_object[0].startswith("E-op") and "[edit 1 of 2" in not_object[0]
    # A refusal of the combined state is about all of them, and names no edit.
    noop = _attempt(Proposal(edits=(_knob("level", 3), _knob("level", 5))))
    assert noop[0].startswith("E-noop: the 2 edits together") and "[edit" not in noop[0]


def test_the_cap_is_enforced_and_one_is_todays_behaviour():
    four = (_knob("extra", 1), _knob("level", 2), _knob("note", "New."),
            {"op": "set_knob", "name": "note", "patch": [{"kind": "append", "content": "More."}]})
    assert isinstance(_attempt(Proposal(edits=four)), Candidate), "the default allows 4"
    assert DEFAULT_MAX_EDITS == 4
    refused = _attempt(Proposal(edits=(*four, _knob("level", 1))))
    assert refused[0].startswith("E-too-many-edits") and "at most 4" in refused[0]
    single_only = _attempt(Proposal(edits=four[:2]), max_edits=1)
    assert single_only[0].startswith("E-too-many-edits")
    assert isinstance(_attempt(Proposal(edit=_knob("level", 1)), max_edits=1), Candidate)


def test_a_single_edit_works_as_before():
    edit = _knob("level", 2)
    assert Proposal(edit=edit).steps == [edit]
    one = Proposal(edits=(edit,))
    assert one.edit == edit and one.edits == () and one.steps == [edit]
    pair = Proposal(edits=(edit, _knob("extra", 1)))
    assert pair.edit == {"edits": [edit, _knob("extra", 1)]}, "the form refusals record"
    assert Proposal(edit=pair.edit).steps == pair.steps, "and it reads back as the list"
    child = _attempt(Proposal(edit=edit))
    assert child.edits == [edit] and child.step_edits == 1


def test_duplicates_are_detected_on_the_combined_state():
    seed = _seed()
    first = _attempt(Proposal(edits=(_knob("level", 3), _knob("extra", 4))), seed)
    # The same point reached by one proposal of two edits in the other order.
    again = _attempt(Proposal(edits=(_knob("extra", 4), _knob("level", 3))), seed, seen=[first.id])
    assert again[0].startswith("E-duplicate") and first.id in again[0]


# ── the loop ──────────────────────────────────────────────────────────────


class Scripted(Optimizer):
    name = "scripted_compound"

    def __init__(self, plan: list[list[Proposal]]) -> None:
        super().__init__(k=1)
        self.plan = list(plan)

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.contexts = getattr(self, "contexts", []) + [ctx]
        return self.plan.pop(0) if self.plan else []


def _run(tmp_path: Path, monkeypatch: Any, plan: list[list[Proposal]], **overrides: Any) -> Loop:
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, **{"epochs": 2, "batch_size": 2, **overrides})
        loop.optimizer = Scripted(plan)
        loop.optimizer.max_edits = loop.config.max_edits_per_proposal
        loop.run()
    return loop


def test_the_loop_records_every_edit_and_credits_every_knob(tmp_path, monkeypatch):
    """The candidate's proposal record, journal and history carry the whole list; the
    knob attribution credits each knob's value in the compound candidate; a resume
    restores which edits were its own."""
    bundle = Proposal(edits=(_knob("level", 3), _knob("extra", 8)), rationale="together")
    loop = _run(tmp_path, monkeypatch, [[bundle], []])
    child = next(c for c in loop.population if c.parent)
    assert child.own_edits() == list(bundle.steps)
    assert loop.proposals[child.id]["edits"] == list(bundle.steps)
    journaled = [r for r in _events(loop, "proposal") if r["id"] == child.id]
    assert journaled and journaled[0]["edits"] == list(bundle.steps)
    assert [r["step_edits"] for r in _events(loop, "candidate") if r["id"] == child.id] == [2]
    attribution = loop.gradient["knob_attribution"]
    assert attribution["level"]["tried"]["3"]["n"] == 1
    assert attribution["extra"]["tried"]["8"]["n"] == 1
    with _patch_mode(TwoLevels):
        resumed = _loop(tmp_path, epochs=2, batch_size=2, resume=True)
        resumed.init()
    again = next(c for c in resumed.population if c.id == child.id)
    assert again.step_edits == 2 and again.own_edits() == list(bundle.steps)


def test_the_loop_refuses_an_unpermitted_edit_inside_a_list_by_index(tmp_path, monkeypatch):
    bundle = Proposal(edits=(_knob("level", 3),
                             {"op": "remove_node", "id": "finish"}))
    loop = _run(tmp_path, monkeypatch, [[bundle], []])
    reasons = [reason for r in loop.rejections for reason in r.reasons]
    assert any(r.startswith("E-policy") and "[edit 2 of 2: remove_node]" in r for r in reasons), reasons
    assert any(r.edit == bundle.edit for r in loop.rejections), "the refusal records the list"


def test_the_run_setting_caps_the_list(tmp_path, monkeypatch):
    bundle = Proposal(edits=(_knob("level", 3), _knob("extra", 8)))
    loop = _run(tmp_path, monkeypatch, [[bundle], []], max_edits_per_proposal=1)
    assert any(reason.startswith("E-too-many-edits") for r in loop.rejections for reason in r.reasons)
    assert all(not c.parent for c in loop.population), "nothing was accepted"
    with _patch_mode(TwoLevels):
        fresh = _loop(tmp_path / "default", max_edits_per_proposal=3)
    assert fresh.optimizer.max_edits == 3, "the loop hands its cap to the optimizer"
    with _patch_mode(TwoLevels), pytest.raises(LoopError, match="max_edits_per_proposal"):
        _loop(tmp_path / "zero", max_edits_per_proposal=0)
    with pytest.raises(profiles.ProfileError, match="max_edits_per_proposal"):
        profiles.normalize({"max_edits_per_proposal": 0})
    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    args = parser.parse_args(["--max-edits-per-proposal", "2", "--fresh-evidence", "0"])
    settings = profiles.normalize({"max_edits_per_proposal": args.max_edits_per_proposal,
                                   "fresh_evidence": args.fresh_evidence})
    assert (settings["max_edits_per_proposal"], settings["fresh_evidence"]) == (2, False)
    assert profiles.normalize({})["max_edits_per_proposal"] == 4


def test_two_proposals_of_one_batch_reaching_one_state_are_one_candidate(tmp_path, monkeypatch):
    first = Proposal(edits=(_knob("level", 3), _knob("extra", 8)))
    second = Proposal(edits=(_knob("extra", 8), _knob("level", 3)))
    loop = _run(tmp_path, monkeypatch, [[first, second], []])
    assert sum(1 for c in loop.population if c.parent) == 1
    assert any(reason.startswith("E-duplicate") for r in loop.rejections for reason in r.reasons)


# ── the autoresearch optimizer ─────────────────────────────────────────────


class _Answer:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.payload)


def test_autoresearch_parses_lists_and_says_it_may_bundle():
    pair = [_knob("level", 3), _knob("extra", 8)]
    client = _Answer({"diagnosis": "d", "proposals": [
        {"edits": pair, "rationale": "cascade"},
        {"edit": _knob("level", 1)},
        {"edits": [_knob("level", 2), "not an edit"]},
    ]})
    optimizer = AutoresearchOptimizer(client)
    parent = _seed()
    proposals = optimizer.propose(ProposalContext(max_edits=4), 3, [parent])
    assert [p.steps for p in proposals] == [pair, [_knob("level", 1)]]
    assert proposals[0].rationale == "cascade"
    assert "up to\n4 edits" in client.prompts[0] or "up to 4 edits" in client.prompts[0].replace("\n", " ")
    assert '"edits": [' in client.prompts[0]
    optimizer.propose(ProposalContext(max_edits=1), 1, [parent])
    assert '"edits": [' not in client.prompts[1], "a run of single edits does not mention bundles"


# ── a structure + knob bundle on LiveMath's skill graph ────────────────────


def _livemath() -> tuple[sf.Applier, Candidate]:
    applier = sf.Applier(sf.seeds()["livemath_skill_graph"])
    return applier, Candidate.from_state(applier.seed_state)


def test_a_new_node_and_its_prompt_knob_in_one_proposal_on_livemath():
    applier, seed = _livemath()
    node = sf._rendered_copy(seed.state, "precision")
    node.update(id="check", system_prompt_template="You check.", writes=["check.txt"],
                produces=["check_fact"])
    insert = {"op": "insert_after", "target": "precision", "node": node}
    prompt = {"op": "set_knob", "name": "check__system_prompt",
              "value": "You check the precision notes for a confirmation cascade."}
    child = applier.attempt(seed, {"edits": [insert, prompt]}, {seed.id})
    assert isinstance(child, Candidate), child
    assert child.state["knobs"]["check__system_prompt"]["value"] == prompt["value"]
    assert "check" in child.state["graph"]["nodes"] and child.own_edits() == [insert, prompt]
    assert sf.check_state(applier, child.state, walker=True) == []
    # The knob does not exist before the node: the other order is refused at edit 1.
    refused = applier.attempt(seed, {"edits": [prompt, insert]}, {seed.id})
    assert isinstance(refused, list) and "[edit 1 of 2: set_knob]" in refused[0], refused


# ── review: per-edit refusals deeper in a list, dead edits, records ─────────


def test_frozen_and_protected_edits_later_in_a_list_are_named():
    seed = _seed()
    frozen = Optimizer().attempt(Proposal(edits=(_knob("level", 3), _knob("extra", 1))), seed,
                                 TwoLevels().invariants(), True, ("extra",), [seed.id])
    assert frozen[0].startswith("E-frozen") and "[edit 2 of 2: set_knob]" in frozen[0], frozen
    applier, live = _livemath()
    text = next(k for k, v in live.state["knobs"].items()
                if (v.get("spec") or {}).get("kind") == "str" and not (v.get("spec") or {}).get("frozen"))
    refused = applier.attempt(live, {"edits": [{"op": "set_knob", "name": text, "value": "New."},
                                               {"op": "remove_node", "id": "finish"}]}, {live.id})
    # A protected node is judged on what the edits leave together, so the refusal
    # names the bundle, not one edit.
    assert isinstance(refused, list) and refused[0].startswith("E-protected"), refused
    assert "[the 2 edits together; from edit 2 (remove_node) on" in refused[0], refused


@pytest.mark.parametrize("edits, dead", [
    ((_knob("extra", 5), _knob("level", 3)), 1),   # extra is already 5
    ((_knob("level", 1), _knob("level", 3)), 1),   # overwritten by the next edit
    ((_knob("level", 3), _knob("extra", 2), _knob("extra", 5)), 2),  # undone
])
def test_an_edit_with_no_effect_in_a_list_is_refused_by_index(edits, dead):
    refused = _attempt(Proposal(edits=edits))
    assert isinstance(refused, list) and refused[0].startswith("E-noop"), refused
    assert f"[edit {dead} of {len(edits)}" in refused[0], refused


def test_pending_candidates_keep_their_compound_edits(tmp_path, monkeypatch):
    """The loop-as-a-graph path writes the optimizer's candidates and installs them in
    another step: a compound candidate keeps its own edits, and a duplicate one is
    refused with the whole list recorded."""
    bundle = Proposal(edits=(_knob("level", 3), _knob("extra", 8)))
    loop = _run(tmp_path, monkeypatch, [[], []], epochs=1)
    child = _attempt(bundle, loop.population[0])
    loop.write_pending([child])
    [again] = loop.read_pending()
    assert again.step_edits == 2 and again.own_edits() == list(bundle.steps)
    assert again.edit_reports == child.edit_reports
    loop.population.append(again)
    loop.step([again])
    assert loop.rejections[-1].edit == {"edits": list(bundle.steps)}
    assert loop.rejections[-1].reasons[0].startswith("E-duplicate")


def test_listed_edits_are_not_clipped_away():
    from refactory_dsh.optim.base import _edit_change_lines, _reasoning_lines

    long = "x" * 5000
    edits = [{"op": "set_knob", "name": f"step{i}_prompt", "value": long} for i in range(4)]
    pairs = [{"child": "c", "parent": None, "edit": edits}]
    text = "\n".join(_edit_change_lines(pairs))
    reasoning = "\n".join(_reasoning_lines([{"id": "c", "edit": edits, "score": 0.5}]))
    for i in range(4):
        assert f'name="step{i}_prompt"' in text and f'name="step{i}_prompt"' in reasoning, i
    assert long not in text, "values are clipped, names are not"


def test_grader_nodes_are_protected_and_grader_side_knobs_hold():
    """Grader status follows a node's id, so a grader node is protected like a protected
    node; a knob a protected or grader node reads or binds holds for every edit."""

    class Graded(TwoLevels):
        name = "graded_levels"

        def protected_nodes(self):  # type: ignore[no-untyped-def]
            return ("finish",)

        def grader_nodes(self):  # type: ignore[no-untyped-def]
            return ("work",)

    mode = Graded()
    assert mode.edit_protected() == ("finish", "work")
    seed = Candidate.from_state(mode.state_dict())

    def attempt(*edits):  # type: ignore[no-untyped-def]
        return Optimizer().attempt(Proposal(edits=edits), seed, mode.invariants(), True, (),
                                   [seed.id], protected=mode.protected_nodes(),
                                   graders=mode.grader_nodes())

    refused = attempt(_knob("note", "Copy the answer into the output."))
    assert refused[0].startswith("E-frozen") and "'work'" in refused[0], refused
    later = attempt({"op": "add_knob", "name": "k9", "spec": {"default": "x", "kind": "str"}},
                    {"op": "remove_knob", "name": "extra"})
    assert later[0].startswith("E-frozen") and "[edit 2 of 2" in later[0], later
    removed = Optimizer().attempt(Proposal(edit={"op": "remove_node", "id": "work"}), seed,
                                  mode.invariants(), True, (), [seed.id],
                                  protected=mode.edit_protected(), graders=mode.grader_nodes())
    assert isinstance(removed, list) and removed[0].startswith("E-protected"), removed


# ── protected nodes are judged on what the bundle leaves ───────────────────


def _replace_decide_prompt(connect: bool) -> dict[str, Any]:
    """The bundle a replayed optimizer call wrote: remove LiveMath's decide_prompt op and
    put a code op in its place that builds decide's prompt from the notes."""
    from refactory_dsh import snippet

    code = ("def main(inp):\n"
            "    parts = [inp['facts'].get('user_prompt', '')]\n"
            "    for name in ('compare_notes.txt', 'precision_notes.txt', 'hypotheses_notes.txt'):\n"
            "        if inp['files'].get(name):\n"
            "            parts.append(inp['files'][name])\n"
            "    parts.append('Choose one option; answer inside <answer>X</answer>.')\n"
            "    return {'facts': {'decide_prompt': '\\n\\n'.join(parts)}}\n")
    node = {"id": "build_decide", "kind": "op", "call": snippet.RUN, "knobs": ["decide_task"],
            "reads": ["compare_notes.txt", "precision_notes.txt", "hypotheses_notes.txt"],
            "required_reads": ["compare_notes.txt"], "produces": ["decide_prompt"],
            "inputs": ["user_prompt"], "code": code}
    edits = [{"op": "remove_node", "id": "decide_prompt"},
             {"op": "add_node", "node": node, "position": {"after": "step_join"}}]
    if connect:
        edits.append({"op": "add_edge", "source": "build_decide", "target": "decide"})
    return {"edits": edits}


@pytest.fixture
def _code_nodes_allowed(monkeypatch):
    """These proposals carry a code op and run nothing: the sandbox probe is assumed."""
    from refactory_dsh import snippet

    monkeypatch.setattr(snippet, "_probe", lambda: (True, "assumed: this test runs no code"))


def test_a_step_replaced_in_place_is_judged_on_what_the_bundle_leaves(_code_nodes_allowed):
    """Removing a step leaves the run, for one edit, able to finish without the protected
    extract_answer; the bundle puts a code op in its place, and only its end state runs,
    so it is accepted (it was refused at edit 1 when protection was checked per edit)."""
    applier, seed = _livemath()
    child = applier.attempt(seed, _replace_decide_prompt(connect=True), {seed.id})
    assert isinstance(child, Candidate), child
    nodes = child.state["graph"]["nodes"]
    assert "decide_prompt" not in nodes and "build_decide" in nodes
    assert {"source": "build_decide", "target": "decide"} in [
        {k: e[k] for k in ("source", "target")} for e in child.state["graph"]["edges"]]
    assert sf.check_state(applier, child.state, walker=False) == []


def test_a_bundle_that_leaves_a_protected_node_skippable_is_still_refused(_code_nodes_allowed):
    """The same bundle without the edge to decide ends the pass at the new op: a run can
    finish without extract_answer, and the refusal names the bundle, not one edit."""
    applier, seed = _livemath()
    refused = applier.attempt(seed, _replace_decide_prompt(connect=False), {seed.id})
    assert isinstance(refused, list) and refused[0].startswith("E-protected"), refused
    assert "'extract_answer'" in refused[0] and "can finish without it" in refused[0], refused
    assert "[the 2 edits together; from edit 1 (remove_node) on" in refused[0], refused
    assert "[edit " not in refused[0], refused


class _Considering(Scripted):
    """Says, with each proposal, what it weighed (an answer's optional ``considered``)."""

    name = "scripted_considering"

    def __init__(self, plan: list[list[Proposal]], considered: str) -> None:
        super().__init__(plan)
        self.considered = considered

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        # The first call's answer, then a later one's: a retry in the same round
        # replaces the optimizer's last answer, never a proposal's own record.
        first = not getattr(self, "contexts", [])
        self.last_diagnosis = "level is too high"
        self.last_considered = self.considered if first else "a later answer"
        return super().propose(ctx, n, parents)


@pytest.mark.parametrize("considered", ["a prompt change, a code step or the setting; the "
                                        "setting is what the evidence names", ""])
def test_what_an_answer_considered_is_journaled_reported_and_shown_back(tmp_path, monkeypatch,
                                                                       considered):
    """An answer's `considered` goes with its proposal: the proposal record and journal
    (that answer's, though a retry in the round answers again), the report, and the next
    request's earlier reasoning. A proposal whose answer gave none records none -- not
    the retry's -- and the report has no such section."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    refused = Proposal(edit={"op": "remove_node", "id": "finish"})  # E-policy: a retry follows
    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=2, batch_size=2)
        loop.optimizer = _Considering([[Proposal(edit=_knob("level", 3)), refused], []],
                                      considered)
        loop.run()
    child = next(c for c in loop.population if c.parent)
    journaled = [r for r in _events(loop, "proposal") if r["id"] == child.id]
    report = (loop.root / "report.md").read_text()
    analysis = "\n".join(p.read_text() for p in (loop.root / "analysis").glob("*.md"))
    later = loop.optimizer.contexts[-1].render()
    if considered:
        assert loop.proposals[child.id]["considered"] == considered
        assert [r.get("considered") for r in journaled] == [considered]
        assert "## What the optimizer weighed" in report and considered in report
        assert "## Considered\n\n" in analysis
        assert f"Considered that round: {considered}" in later
    else:
        assert "considered" not in loop.proposals[child.id] and journaled
        assert all("considered" not in r for r in journaled)
        assert "What the optimizer weighed" not in report and "a later answer" not in report
        assert "Considered that round" not in later


def test_e_protected_names_the_edit_no_later_edit_repairs(_code_nodes_allowed):
    """A bundle that breaks a protected node at edit 1, mends it at edit 3 and breaks it
    again at edit 4 is refused naming edit 4: from there on no edit restores it."""
    applier, seed = _livemath()
    bundle = _replace_decide_prompt(connect=True)
    bundle["edits"].append({"op": "remove_edge", "source": "build_decide", "target": "decide"})
    refused = applier.attempt(seed, bundle, {seed.id})
    assert isinstance(refused, list) and refused[0].startswith("E-protected"), refused
    assert "[the 4 edits together; from edit 4 (remove_edge) on" in refused[0], refused


def test_a_slot_stranded_by_a_later_round_is_refused(_code_nodes_allowed):
    """Round 1 puts a fact in decide_task, which decide_prompt fills (accepted). Round 2
    replaces decide_prompt with a code op that reads decide_task as written: the slot
    the parent filled would now reach decide as braces, so V20 refuses it."""
    applier, seed = _livemath()
    value = seed.state["knobs"]["decide_task"]["value"] + " Recall {{user_prompt}}."
    child = applier.attempt(seed, {"op": "set_knob", "name": "decide_task", "value": value},
                            {seed.id})
    assert isinstance(child, Candidate), child
    refused = applier.attempt(child, _replace_decide_prompt(connect=True), {seed.id, child.id})
    assert isinstance(refused, list) and any(
        r.startswith("V20") and "'decide_task'" in r and "{{user_prompt}}" in r
        for r in refused), refused
