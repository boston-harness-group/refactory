"""The pipeline view (`nn.pipeline_view`): one line per step, in the core operators' words.

The request opens its structure part with it, and a candidate's record and the next
request say what its edit built as the view's new and changed lines, so the proposer reads
one vocabulary everywhere, never a hand-written description. These tests hold the view to
the graph: everything a model step takes in is in its ``sees``, every gate and fork and
edge is shown, each ``loop``/``route``/``parallel``/``sees`` line names the handles an
operator takes (applying them gives the same line back), and the S4 and S1 views fit in 2k
characters. ``sees`` takes single fact or file names, which the request now says.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.graph import Graph
from refactory_dsh.nn import validate_state
from refactory_dsh.nn.validate import consumed
from refactory_dsh.optim import Candidate, base


def _mode(name: str):
    _package, mode = registry.resolve(Path("."), name)
    return mode()


def _lines(state, protected=()):
    return {line.key: line.text for line in pv.view_lines(state, protected)}


# ── the view ────────────────────────────────────────────────────────────────


def test_the_s4_and_s1_views_say_each_step_in_one_line_under_2k():
    s4 = _mode("livemath/skill_graph")
    text = pv.render(s4.state_dict(), list(s4.edit_protected()))
    assert len(text) < 2000, len(text)
    lines = text.splitlines()
    assert lines[0] == "per item (data -> current_item.json):"
    assert lines[1].startswith("  compare agent  sees compare_prompt@prepare(current_item.json@data)"
                               "  told compare__system_prompt \"")
    assert "compare_task? \"" in lines[1] and lines[1].endswith("-> compare_notes.txt")
    assert lines[2] == "  parallel {"
    assert lines[3].startswith("    precision agent  sees "
                               "precision_prompt@notes_prompts(compare_notes.txt@compare)")
    assert lines[5] == "  } -> step_join"
    assert lines[6].startswith("  decide agent  sees decide_prompt@decide_prompt("
                               "compare_notes.txt@compare, hypotheses_notes.txt@hypotheses, "
                               "precision_notes.txt@precision)")
    assert lines[7] == ("  [protected] extract_answer op  sees answer.txt@decide, "
                        "current_item.json@data  -> answer_label, answer_valid")
    assert lines[8] == "[protected] finish op  -> result.json, finished"
    s1 = _mode("livemath/solver")
    text1 = pv.render(s1.state_dict(), list(s1.edit_protected()))
    assert len(text1) < 2000
    assert "solve agent  sees skill_section@format_prompt(current_item.json@data), " \
        "solver_prompt@format_prompt(current_item.json@data)  told solve__system_prompt* \"" \
        in text1, text1  # * frozen


def _model_steps(graph: Graph, view) -> set[str]:
    return {n for n, node in graph.nodes.items()
            if node.kind == "agent" or n in view.code}


@pytest.mark.parametrize("seed", sorted(sf.seeds()))
def test_every_input_gate_fork_and_edge_is_shown_on_seeds_and_random_edits(seed):
    """On each seed graph and 40 operator draws from it (those the validator accepts): each
    model or code step has a
    line (a judge's is inside its gate's), with every input it takes in by name; every
    gate is a loop or route line and every fork a parallel block; along every forward
    edge between shown steps the target's line comes after the source's."""
    state = sf.Applier(sf.seeds()[seed]).seed_state
    rng = random.Random(f"view:{seed}")
    states = [state]
    for _ in range(40):
        edit = nn_edits.random_edit(state, preset="broad", rng=rng, operators=True)
        if edit is None or nn_edits.precheck(edit, state) is not None:
            continue
        try:
            after = nn_edits.apply(edit, state)
        except nn_edits.EditError:
            continue
        if not validate_state(after):  # a graph the validator would refuse is never shown
            states.append(after)
    for current in states:
        view = pv._View(current, ())
        lines = view.lines()
        text = {line.key: line.text for line in lines}
        order = {line.key: index for index, line in enumerate(lines)}
        graph = view.graph
        for node_id in _model_steps(graph, view):
            gate = next((g for g, j in view.judges.items() if j == node_id), None)
            line = text.get(gate) if gate else text.get(node_id)
            assert line is not None, (seed, node_id, list(text))
            for _kind, name in consumed(graph.nodes[node_id], view.values):
                assert f"{name}@" in line, (seed, node_id, name, line)
        for node_id, node in graph.nodes.items():
            if node.kind == "gate":
                assert re.match(r"(loop|route)\(", text.get(node_id, "")), (seed, node_id)
            if node.kind == "fork":
                assert text.get(node_id) == "parallel {", (seed, node_id)
        # Along every forward edge from a step the walk reaches from the start (a graph an
        # edit left in pieces has no order between its pieces).
        reached = set(view._reach(graph.start))
        for edge in graph.edges:
            a, b = edge.source, edge.target
            if a in order and b in order and a in reached and (a, b) not in view.back:
                assert order[b] > order[a], (seed, a, b)


def test_a_judge_s_loop_line_names_the_handles_a_loop_takes():
    """Round trip: the step and cap a loop line shows, put back into a loop edit with the
    same text, give the same line."""
    s1 = _mode("livemath/solver").state_dict()
    edit = {"op": "loop", "step": "solve", "judge": "agent", "text": "Check it.",
            "max_iterations": 3}
    line = _lines(nn_edits.apply(edit, s1))["solve_check"]
    found = re.match(r"loop\((\w+), judge (\w+) .*, max (\d) -> (\d) runs, knob (\w+)\)$", line)
    assert found, line
    step, judge, cap, runs, knob = found.groups()
    assert (step, judge, int(cap), int(runs)) == ("solve", "solve_check_judge", 3, 4)
    again = {"op": "loop", "step": step, "judge": "agent", "text": "Check it.",
             "max_iterations": int(cap)}
    assert _lines(nn_edits.apply(again, s1))["solve_check"] == line
    assert knob == "solve_check__max_iterations"


def test_a_route_line_names_the_handles_a_route_takes():
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    graph = Graph.from_state(state["graph"])
    anchor = next(n for n, node in graph.nodes.items() if node.kind == "agent"
                  and len(graph.out_edges(n)) == 1)
    nxt = graph.out_edges(anchor)[0].target
    edit = {"op": "route", "after": anchor, "judge": "agent", "text": "Choose.",
            "to": {"proceed": nxt, "revise": {"kind": "agent", "text": "Revise it.",
                                              "sees": ["task"]}}}
    after = nn_edits.apply(edit, state)
    line = next(text for text in _lines(after).values() if text.startswith("route("))
    found = re.match(r"route\(after (\w+), judge \w+ .*, \{(.*)\}\)$", line)
    assert found and found.group(1) == anchor, line
    labels = dict(part.split(": ") for part in found.group(2).split(", "))
    assert labels["proceed"] == nxt and set(labels) == {"proceed", "revise"}
    # A label names the step it leads to as the view shows it, not a collect step before it.
    shown = _lines(after)
    assert labels["revise"] in shown and shown[labels["revise"]].startswith(
        f"{labels['revise']} agent"), (labels, list(shown))


def test_a_branch_s_sees_line_names_exactly_what_it_was_given():
    """The reviewer's case and its round trip: on S4 a branch that sees ``user_prompt``
    takes in that alone, and its line's names, put back into ``sees``, give it again."""
    s4 = _mode("livemath/skill_graph").state_dict()
    edit = {"op": "branch", "fork": "step_fork", "kind": "agent", "text": "{{user_prompt}}",
            "sees": ["user_prompt"], "id": "fresh"}
    after = nn_edits.apply(edit, s4)
    graph = Graph.from_state(after["graph"])
    values = {k: e["value"] for k, e in after["knobs"].items()}
    assert consumed(graph.nodes["fresh"], values) == {("fact", "user_prompt")}
    line = _lines(after)["fresh"]
    assert line.startswith("fresh agent  sees user_prompt@prepare(current_item.json@data)  told")
    names = re.findall(r"(\w[\w.]*)@", line.split("  sees ", 1)[1].split("  told", 1)[0])
    assert names[0] == "user_prompt"
    again = nn_edits.apply({**edit, "sees": [names[0]]}, s4)
    assert _lines(again)["fresh"] == line
    wide = nn_edits.apply({**edit, "sees": ["task"]}, s4)
    assert "compare_prompt@prepare" in _lines(wide)["fresh"]


def test_sees_takes_a_file_name_on_a_nested_fork_and_lists_names_when_refused():
    nested = sf.Applier(sf.seeds()["fuzz_nested_fork"]).seed_state
    graph = Graph.from_state(nested["graph"])
    from refactory_dsh.nn import core_ops

    fork, files = next((f, sorted(n for k, n in core_ops._available(graph)[f] if k == "file"))
                       for f, node in sorted(graph.nodes.items()) if node.kind == "fork"
                       and any(k == "file" for k, _n in core_ops._available(graph).get(f, ())))
    edit = {"op": "branch", "fork": fork, "kind": "agent", "sees": [files[0]], "id": "reader"}
    after = nn_edits.apply(edit, nested)
    assert f"{files[0]}@" in _lines(after)["reader"]
    with pytest.raises(nn_edits.EditError) as refused:
        nn_edits.apply({**edit, "sees": ["no_such_input"]}, nested)
    message = str(refused.value)
    assert "'sees' names 'no_such_input'" in message and f"there: " in message
    assert files[0] in message.split("there: ", 1)[1]


def test_the_catalogue_says_sees_takes_single_names():
    lines = {op["op"]: json_text(op) for op in nn_edits.catalogue("broad")}
    assert "single fact or file names" in lines["rewire"]
    assert "single fact or file names" in lines["branch"]


def json_text(op) -> str:
    import json

    return json.dumps(op)


# ── as built ────────────────────────────────────────────────────────────────


def test_what_an_edit_built_is_its_new_and_changed_view_lines():
    s1 = _mode("livemath/solver").state_dict()
    looped = nn_edits.apply({"op": "loop", "step": "solve", "judge": "agent"}, s1)
    built = pv.changed_lines(s1, looped)
    assert len(built) == 1 and built[0].startswith("+ loop(solve, judge solve_check_judge (sees ")
    assert built[0].endswith("max 2 -> 3 runs, knob solve_check__max_iterations)")
    lowered = nn_edits.apply({"op": "set_knob", "name": "solve_check__max_iterations",
                              "value": 1}, looped)
    changed = pv.changed_lines(looped, lowered)
    assert len(changed) == 1 and changed[0].startswith("~ loop(solve,")
    assert "max 1 -> 2 runs" in changed[0]
    s4 = _mode("livemath/skill_graph").state_dict()
    branched = nn_edits.apply({"op": "branch", "fork": "step_fork", "kind": "agent",
                               "sees": ["task"], "id": "independent"}, s4)
    built = pv.changed_lines(s4, branched)
    assert any(line.startswith("+ independent agent  sees compare_prompt@prepare(") and
               "current_item.json@data" in line and "user_prompt@prepare(" in line
               for line in built), built
    assert any(line.startswith("~ decide agent") and "independent@independent" in line
               for line in built), built
    draft = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    step = next(n for n in sorted(draft["graph"]["nodes"])
                if nn_edits.precheck({"op": "remove", "step": n}, draft) is None)
    assert f"- removed {step}" in pv.changed_lines(
        draft, nn_edits.apply({"op": "remove", "step": step}, draft))
    patched = nn_edits.apply({"op": "set_knob", "name": "decide_task", "value": "Decide."}, s4)
    assert all(line.startswith("~ decide agent") for line in pv.changed_lines(s4, patched))


def test_the_request_shows_the_view_first_and_as_built_beside_the_words():
    pair = {"child": "c1", "parent": "p0", "edit": [{"op": "loop", "step": "solve"}],
            "flow_change": "the judge sends solve back once",
            "wiring": ["+ loop(solve, judge j, max 2 -> 3 runs)"]}
    lines = base._edit_change_lines([pair])
    said = lines.index("Flow change, as proposed: the judge sends solve back once")
    assert lines[said + 1].startswith("As built (its pipeline lines")
    assert lines[said + 2:said + 5] == ["```text", "+ loop(solve, judge j, max 2 -> 3 runs)",
                                        "```"]
    text = "\n".join(base._reasoning_lines([{**pair, "id": "c1", "epoch": 1}]))
    assert "Flow change, as proposed: the judge sends solve back once\nAs built" in text
    many = base._as_built_lines({"wiring": [f"+ line {i}" for i in range(20)]})
    assert f"(and {20 - base.AS_BUILT_LINES} more line(s))" in many


def test_the_loop_s_request_opens_with_the_view_and_shows_what_was_built(tmp_path):
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig

    loop = strategies.create("default", "livemath/solver", LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path,
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    edit = {"op": "loop", "step": "solve", "judge": "agent"}
    child = Candidate.from_state(nn_edits.apply(edit, seed.state), parent=seed.id, edits=[edit])
    child.evaluated, child.score = True, 0.5
    seed.evaluated, seed.score = True, 0.4
    loop.population += [seed, child]
    loop._editing_parent = child.id
    loop.proposals[child.id] = {"id": child.id, "parent": seed.id, "epoch": 1,
                                "origin": "optimizer",
                                "flow_change": "solve is sent back once at most"}
    text = loop._context().render()
    pipeline = text.split("## The pipeline you may edit", 1)[1].split("Nodes:", 1)[0]
    assert pipeline.index(base.PIPELINE_VIEW_LEAD) < pipeline.index("```text\nper item (data")
    assert "loop(solve, judge solve_check_judge" in pipeline  # the edited candidate's view
    for section in ("## What the last edit changed", "## Your earlier reasoning"):
        body = text.split(section, 1)[1].split("\n## ", 1)[0]
        assert "Flow change, as proposed: solve is sent back once at most" in body, section
        assert "+ loop(solve, judge solve_check_judge" in body, section
        assert "max 2 -> 3 runs" in body, section


def test_a_round_records_what_was_built_in_the_proposal_record(tmp_path, monkeypatch):
    from refactory_dsh.loop import Loop
    from refactory_dsh.optim import Optimizer, Proposal
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_train_minibatch import _events
    from test_val_gate import _loop

    edit = {"op": "loop", "step": "work", "judge": "loop_ops:always_proceed",
            "max_iterations": 1}

    class Scripted(Optimizer):
        name = "scripted_view"

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
    assert built == ["+ loop(work, judge loop_ops:always_proceed, max 1 -> 2 runs, knob "
                     "work_check__max_iterations)"], built
    journaled = [r for r in _events(loop, "proposal") if r["id"] == child.id]
    assert journaled and journaled[0]["wiring"] == built
