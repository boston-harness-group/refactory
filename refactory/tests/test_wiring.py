"""What an edit built, read from the expanded graph (`nn.wiring`).

A proposal's prose (``flow_change``) can say one thing while its edit's expansion does
another: a loop left at its default cap described as "one retry", a branch that sees the
task described as seeing the question alone. The candidate's record and the next request
carry the wiring read from the candidate's graph against its parent's, so the prose is
never the only description. These tests hold `as_built` to the graph on several shapes,
and the request to showing it beside the proposal's own words.
"""

from __future__ import annotations

from pathlib import Path

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import wiring
from refactory_dsh.optim import Candidate, base


def _mode_state(name: str) -> dict:
    _package, mode = registry.resolve(Path("."), name)
    return mode().state_dict()


def test_a_loop_s_line_states_its_effective_cap_from_the_graph():
    solver = _mode_state("livemath/solver")
    looped = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent"}, solver)
    lines = wiring.as_built(solver, looped)
    assert ("new gate solve_check on solve_check_judge.txt (written by solve_check_judge): "
            "sends work back to solve at most 2 time(s) (knob solve_check__max_iterations = 2), "
            "so solve runs at most 3 times per item; then the run goes on with the last "
            "attempt") in lines, lines
    # The judge's real inputs: what solve read, and solve's reply through a collect step.
    assert ("new agent step solve_check_judge: takes in skill_section (format_prompt), "
            "answer.txt (solve, via solve_check_judge_inputs), solver_prompt (format_prompt); "
            "writes solve_check_judge.txt; read by solve_check") in lines, lines
    assert "new no-op step solve_done after solve_check" in lines
    one = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent",
                          "max_iterations": 1}, solver)
    assert any("at most 1 time(s)" in line and "runs at most 2 times" in line
               for line in wiring.as_built(solver, one))
    # A cap changed later by its knob: the gate's line again, with the value it had.
    lowered = nn_edits.apply({"op": "set_knob", "name": "solve_check__max_iterations",
                              "value": 1}, looped)
    assert wiring.as_built(looped, lowered) == [
        "gate solve_check on solve_check_judge.txt (written by solve_check_judge): sends "
        "work back to solve at most 1 time(s) (knob solve_check__max_iterations = 1, was 2), "
        "so solve runs at most 2 times per item; then the run goes on with the last attempt"]


def test_the_cap_comes_from_the_graph_not_the_edit_text():
    """The written-out loop carries its cap on the gate node: the line reads the gate,
    whatever the edit is called or says."""
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    gate = {"id": "review_gate", "kind": "gate", "evaluator": "refactory_dsh.snippet:gate",
            "code": "def main(inp):\n    return {'verdict': 'proceed', 'reason': ''}\n",
            "max_iterations": 3}
    after = nn_edits.apply({"op": "loop", "target": "review", "gate": gate}, state)
    lines = wiring.as_built(state, after)
    assert any(line.startswith("new gate review_gate on its own check: sends work back to "
                               "review at most 3 time(s)") and "review runs at most 4 times"
               in line for line in lines), lines


def test_a_task_branch_s_line_names_every_input_with_its_maker():
    s4 = _mode_state("livemath/skill_graph")
    after = nn_edits.apply({"op": "branch", "fork": "step_fork", "kind": "agent",
                            "text": "{{user_prompt}}", "sees": ["task"], "id": "independent"},
                           s4)
    lines = wiring.as_built(s4, after)
    assert ("new agent step independent, a branch of fork step_fork: takes in compare_prompt "
            "(prepare), current_item.json (data, via independent_inputs), user_prompt (prepare); "
            "writes independent.txt, makes fact independent; read by decide_prompt") in lines, lines
    assert any(line.startswith("step decide_prompt now takes in") and
               "(new: independent (independent))" in line for line in lines), lines


def test_nested_forks_rewires_code_steps_and_removals_are_read_the_same_way():
    nested = sf.Applier(sf.seeds()["fuzz_nested_fork"]).seed_state
    forks = sorted(n for n, x in nested["graph"]["nodes"].items() if x.get("kind") == "fork")
    inner = forks[-1]
    branched = nn_edits.apply({"op": "branch", "fork": inner, "kind": "agent"}, nested)
    assert any(f", a branch of fork {inner}:" in line
               for line in wiring.as_built(nested, branched))
    s4 = _mode_state("livemath/skill_graph")
    rewired = nn_edits.apply({"op": "rewire", "step": "compare", "sees": ["task"]}, s4)
    assert wiring.as_built(s4, rewired) == [
        "step compare now takes in current_item.json (data, via compare_inputs), compare_prompt "
        "(prepare), user_prompt (prepare) (new: current_item.json (data, via compare_inputs), "
        "user_prompt (prepare))"]
    flow = sf.Applier(sf.seeds()["fuzz_dataflow"]).seed_state
    coded = nn_edits.apply({"op": "insert", "after": "summarise", "kind": "code", "id": "tally"},
                           flow)
    assert any(line.startswith("new code step tally: takes in") for line in
               wiring.as_built(flow, coded))
    draft = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    step = next(n for n in sorted(draft["graph"]["nodes"])
                if nn_edits.precheck({"op": "remove", "step": n}, draft) is None)
    removed = nn_edits.apply({"op": "remove", "step": step}, draft)
    assert f"removed step {step}" in wiring.as_built(draft, removed)


def test_nothing_built_says_nothing():
    s4 = _mode_state("livemath/skill_graph")
    patched = nn_edits.apply({"op": "set_knob", "name": "decide_task", "value": "Decide."}, s4)
    assert wiring.as_built(s4, patched) == []
    assert wiring.as_built({"graph": {"nodes": {"x": {}}}}, s4) == []


def test_the_request_shows_the_wiring_beside_the_proposal_s_own_words():
    pair = {"child": "c1", "parent": "p0", "edit": [{"op": "loop", "step": "solve"}],
            "flow_change": "the judge sends solve back once",
            "wiring": ["new gate g: sends work back to solve at most 2 time(s), so solve runs "
                       "at most 3 times per item"]}
    lines = base._edit_change_lines([pair])
    said = lines.index("Flow change, as proposed: the judge sends solve back once")
    assert lines[said + 1].startswith("As built (read from the candidate's graph; where the "
                                      "proposal's words say otherwise, this is what ran)")
    assert lines[said + 2] == ("- new gate g: sends work back to solve at most 2 time(s), so "
                               "solve runs at most 3 times per item")
    text = "\n".join(base._reasoning_lines([{**pair, "id": "c1", "epoch": 1}]))
    assert "Flow change, as proposed: the judge sends solve back once\nAs built" in text
    many = base._as_built_lines({"wiring": [f"line {i}" for i in range(20)]})
    assert many[-1] == f"- (and {20 - wiring.WIRING_LINES} more wiring line(s))"


def test_the_loop_records_the_wiring_and_hands_it_to_the_next_request(tmp_path):
    """Through a run: the edited candidate's child carries its wiring in the proposal
    record, and the next request's two views of it show the cap the graph has."""
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig

    loop = strategies.create("default", "livemath/solver", LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path,
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    child_state = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent"}, seed.state)
    child = Candidate.from_state(child_state, parent=seed.id,
                                 edits=[{"op": "loop", "step": "solve", "judge": "agent"}])
    child.evaluated, child.score = True, 0.5
    seed.evaluated, seed.score = True, 0.4
    loop.population += [seed, child]
    loop._editing_parent = child.id
    loop.proposals[child.id] = {"id": child.id, "parent": seed.id, "epoch": 1, "origin": "optimizer",
                                "flow_change": "solve is sent back once at most"}
    text = loop._context().render()
    for section in ("## What the last edit changed", "## Your earlier reasoning"):
        body = text.split(section, 1)[1].split("\n## ", 1)[0]
        assert "Flow change, as proposed: solve is sent back once at most" in body, section
        assert "sends work back to solve at most 2 time(s)" in body, section
        assert "so solve runs at most 3 times per item" in body, section


def test_a_round_records_the_wiring_in_the_proposal_record(tmp_path, monkeypatch):
    """Through a real round: the proposal record and its journal entry carry what the
    edit built, with the cap the edit set."""
    from refactory_dsh.loop import Loop
    from refactory_dsh.optim import Optimizer, Proposal
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_train_minibatch import _events
    from test_val_gate import _loop

    edit = {"op": "loop", "step": "work", "judge": "loop_ops:always_proceed",
            "max_iterations": 1}

    class Scripted(Optimizer):
        name = "scripted_wiring"

        def __init__(self) -> None:
            super().__init__(k=1)
            self.plan = [[Proposal(edit=edit, flow_change="work is checked once")], []]

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return self.plan.pop(0) if self.plan else []

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=2, batch_size=1, allowed_ops=(), edit_preset="broad")
        loop.optimizer = Scripted()
        loop.run()
    child = next(c for c in loop.population if c.parent)
    built = loop.proposals[child.id]["wiring"]
    assert any("sends work back to work at most 1 time(s)" in line
               and "work runs at most 2 times per item" in line for line in built), built
    journaled = [r for r in _events(loop, "proposal") if r["id"] == child.id]
    assert journaled and journaled[0]["wiring"] == built
