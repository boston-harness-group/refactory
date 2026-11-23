"""The proposer can build what the graph runs: gaps an audit of the step-making operators
found (2026-11-23), each closed on the path the operators already share.

- A ``route`` inside a loop (every SpreadsheetBench step is inside turn_gate -> frame's
  reloop) may lead on: "leads back" is judged on the forward flow, retries left out.
- A step's own collect step (folded into the step in the view) goes with it: ``merge``
  sees through it, and a route way to the step enters through it.
- A copy of a sealed tool step stays sealed, with its files in its workspace; the route's
  default ``revise`` way is such a copy, as is its judge's role, timeout and route.
- A ``map``'s combine may have tools: its runs' results reach it from a collect step.
- ``branch`` takes 'tools'; ``merge`` keeps sealed steps' tools; core operators take
  Claude Code's tool names as ``add_node`` does; a map's runs take their neighbour's
  timeout; the route line names every operator that takes 'route'.
"""
from __future__ import annotations

import sys

import item_files_ops as ifo
from refactory_dsh.loop import _new_wiring_warnings
from refactory_dsh.nn import core_ops
from refactory_dsh.nn.validate import validate_state
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.workspace import is_sealed
from refactory_packages.spreadsheetbench import mode as sb_mode

T = ["bash"]
LOOK = {"op": "insert", "before": "solve", "id": "look", "kind": "agent", "text": "Look.",
        "tools": T, "to": "solve"}


def _attempt(mode, edits: list[dict], parent: Candidate | None = None):
    optimizer = Optimizer()
    optimizer.sealed_tools = (True, "")
    parent = parent or Candidate.from_state(mode.state_dict())
    return optimizer.attempt(Proposal(edits=edits), parent, mode.invariants(), True, (),
                             protected=mode.edit_protected(), record=False)


def _nodes(out) -> dict:
    assert isinstance(out, Candidate), out
    return out.state["graph"]["nodes"]


def _edges(out) -> set[tuple]:
    return {(e["source"], e["target"], e["condition"]) for e in out.state["graph"]["edges"]}


# ── H1: a route inside a loop ─────────────────────────────────────────────


def test_a_route_after_a_step_inside_a_loop_leads_on():
    """SpreadsheetBench: solve runs inside turn_gate's reloop to frame. A route after it
    goes on to run_turn (the default 'proceed'), with a judge, a code judge, or ways named;
    a way back to frame is still refused."""
    sb = sb_mode.Solver()
    nodes = _nodes(_attempt(sb, [{"op": "route", "after": "solve", "judge": "agent"}]))
    assert "solve_route" in nodes and "solve_revise" in nodes
    assert isinstance(_attempt(sb, [{"op": "route", "after": "solve", "judge": "code"}]),
                      Candidate)
    back = _attempt(sb, [{"op": "route", "after": "solve", "judge": "agent",
                          "to": {"proceed": "run_turn", "back": "frame"}}])
    assert isinstance(back, list) and "route 'back' leads back to 'frame'" in back[0], back


ROUTE_PROMPT = "Choose the way."
LOOP_PROMPT = "Check the answer."
REDO_PROMPT = "Redo it."

#: A stand-in DSH CLI answering each step by its prompt: the route's judge sends the first
#: item it sees to 'revise' and the rest on; the loop's judge accepts.
ROUTE_AGENT = """\
import pathlib, sys
prompt = sys.stdin.read()
log = pathlib.Path(%r)
seen = log.read_text().splitlines() if log.exists() else []
if %r in prompt:
    answer = 'revise' if not any(l == 'route' for l in seen) else 'proceed'
    log.write_text('\\n'.join(seen + ['route']))
    print(answer + ': because')
elif %r in prompt:
    print('proceed: fine')
elif %r in prompt:
    log.write_text('\\n'.join(seen + ['redo']))
    print('the redone answer')
else:
    print('an answer')
"""


def test_walker_runs_a_route_inside_a_loop(tmp_path):
    """[walker] A loop on solve, then a route after solve inside it (refused before): the
    route's judge sends one item to the new 'revise' step, which runs, and every item
    goes on through the loop's judge to the end of its pass."""
    from refactory_dsh.model import DshModelClient

    mode = ifo.AgentPass()
    out = _attempt(mode, [{"op": "loop", "step": "solve", "judge": "agent", "text": LOOP_PROMPT},
                          {"op": "route", "after": "solve", "judge": "agent", "text": ROUTE_PROMPT,
                           "to": {"proceed": "solve_check_judge",
                                  "revise": {"kind": "agent", "text": REDO_PROMPT,
                                             "takes_over": "solve"}}}])
    nodes = _nodes(out)
    assert {"solve_route", "solve_check"} <= set(nodes)
    mode.load_state_dict(out.state)
    log = tmp_path / "calls.log"
    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n"
                      + ROUTE_AGENT % (str(log), ROUTE_PROMPT, LOOP_PROMPT, REDO_PROMPT))
    script.chmod(0o755)
    client = DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)
    result = mode.run(tmp_path / "walker", model=client)
    assert result.exit_ok, result.error
    calls = log.read_text().splitlines()
    assert calls.count("route") == 2 and calls.count("redo") == 1, calls
    assert len(result.facts["item_scores"]) == 2


# ── H2: a step's own collect step goes with it ────────────────────────────


#: solve -> s2_inputs -> s2 -> s3_inputs -> s3: each new step owns the collect step in
#: front of it (it alone reads what that hands on).
THREE = [{"op": "insert", "after": "solve", "kind": "agent", "text": "Polish.", "id": "s2",
          "takes_over": "solve"},
         {"op": "insert", "after": "s2", "kind": "agent", "text": "Check.", "id": "s3",
          "takes_over": "s2", "sees": ["s2"]}]


def test_merge_sees_through_a_step_s_own_collect_step():
    mode = ifo.AgentPass()
    out = _attempt(mode, [*THREE, {"op": "merge", "steps": ["solve", "s2"]}])
    nodes = _nodes(out)
    assert "s2_inputs" not in nodes and "solve" not in nodes and "s2" not in nodes
    assert ("solve_s2", "s3_inputs", None) in _edges(out)
    assert validate_state(out.state) == []
    # The first step's collect step too: the merged step sees what it handed on.
    out = _attempt(mode, [*THREE, {"op": "merge", "steps": ["s2", "s3"]}])
    nodes = _nodes(out)
    assert nodes["s2_s3_inputs"]["reads"] == ["solve.txt"]
    assert "s2_inputs" not in nodes and "s3_inputs" not in nodes


def test_a_route_way_to_a_step_enters_through_its_collect_step():
    mode = ifo.AgentPass()
    parent = _attempt(mode, THREE[:1])
    out = _attempt(mode, [{"op": "route", "after": "solve", "judge": "agent",
                           "to": {"keep": "s2", "alt": {"kind": "agent", "text": "x",
                                                        "takes_over": "solve"}}}],
                   parent=parent)
    assert ("solve_route", "s2_inputs", "keep") in _edges(out)
    assert not any(e[1] == "s2" and e[0] == "solve_route" for e in _edges(out))


# ── H3, H4, M1: copies of sealed steps; the route's judge and revise way ─────


def test_a_copy_of_a_sealed_tool_step_is_sealed_with_its_files():
    sb = sb_mode.Solver()
    look = {"op": "insert", "after": "frame", "id": "look", "kind": "agent", "text": "Look.",
            "tools": T, "to": "solve"}
    nodes = _nodes(_attempt(sb, [look, {"op": "insert", "after": "look", "id": "look_copy",
                                        "to": "solve"}]))
    assert is_sealed(nodes["look_copy"])
    assert nodes["look_copy"]["reads"] == nodes["look"]["reads"] == ["history.json",
                                                                      "item/input.xlsx"]
    assert "look_copy_inputs" not in nodes
    nodes = _nodes(_attempt(ifo.AgentPass(), [LOOK, {"op": "parallelize", "steps": ["look"],
                                                     "copies": 1}]))
    assert is_sealed(nodes["look_copy1"]) and "look_copy1_inputs" not in nodes


def test_the_route_s_default_revise_way_and_judge_take_the_step_s_settings():
    nodes = _nodes(_attempt(ifo.AgentPass(), [LOOK,
                                              {"op": "route", "after": "look",
                                               "judge": "agent"}]))
    assert is_sealed(nodes["look_revise"]) and nodes["look_revise"]["tools"] == T
    sb = sb_mode.Solver()
    nodes = _nodes(_attempt(sb, [{"op": "route", "after": "solve", "judge": "agent"}]))
    # The judge and the revise way take solve's role, as a loop's judge does.
    assert nodes["solve_route_judge"]["role"] == nodes["solve"]["role"] == "solver"
    assert nodes["solve_revise"]["role"] == "solver"


# ── H5: a map's combine with tools ────────────────────────────────────────


def test_a_map_s_combine_may_have_tools():
    sb = sb_mode.Solver()
    out = _attempt(sb, [{"op": "map", "after": "frame", "over": "solver_prompt",
                         "text": "Do one.", "tools": T, "to": "solve",
                         "combine": {"kind": "agent", "text": "Combine.", "tools": T}}])
    nodes = _nodes(out)
    combine = nodes["frame_map_combine"]
    assert is_sealed(combine)
    assert not any(r.startswith("frame_map_") for r in combine["reads"]), combine["reads"]
    gather = nodes["frame_map_combine_inputs"]
    assert gather["reads"] == [f"frame_map_{i}.txt" for i in range(1, 5)]
    assert ("frame_map_combine_inputs", "frame_map_combine", None) in _edges(out)


# ── M2: a taken-over step left running ────────────────────────────────────


def test_a_take_over_that_leaves_its_step_unread_is_said_before_it_runs():
    """SpreadsheetBench: a new step takes solve over and sees only the task; solve still
    runs, its reply read by nothing. Accepted (a warning, never a refusal), and the
    warning the proposer is shown names the take-over."""
    sb = sb_mode.Solver()
    out = _attempt(sb, [{"op": "insert", "after": "solve", "kind": "agent", "id": "redo",
                         "text": "Answer anew.", "takes_over": "solve", "sees": ["task"]}])
    assert isinstance(out, Candidate), out
    [warning] = _new_wiring_warnings(out.state, sb.state_dict())
    assert warning.startswith("W-unread-output [solve]")
    assert "now written by 'redo', which took it over" in warning


# ── M3, tools lists, M4 ───────────────────────────────────────────────────


def test_branch_takes_tools_and_merge_keeps_a_sealed_step_s_tools():
    sb = sb_mode.Solver()
    pair = {"op": "parallelize", "after": "frame", "new": [
        {"kind": "agent", "text": "A", "to": "solve", "id": "a1"},
        {"kind": "agent", "text": "B", "to": "solve", "id": "b1"}]}
    nodes = _nodes(_attempt(sb, [pair, {"op": "branch", "fork": "frame_fork", "kind": "agent",
                                        "text": "C", "tools": T, "to": "solve", "id": "c1"}]))
    assert is_sealed(nodes["c1"])
    nodes = _nodes(_attempt(ifo.AgentPass(), [
        LOOK, {"op": "insert", "after": "look", "kind": "agent", "text": "L2", "id": "look2",
               "takes_over": "look"},
        {"op": "merge", "steps": ["look", "look2"], "text": "both"}]))
    assert is_sealed(nodes["look_look2"]) and nodes["look_look2"]["tools"] == T


def test_core_operators_take_claude_code_s_tools_as_add_node_does():
    nodes = _nodes(_attempt(ifo.AgentPass(), [{**LOOK, "tools": ["Read", "Bash"]}]))
    assert is_sealed(nodes["look"]) and nodes["look"]["tools"] == ["Read", "Bash"]
    refused = _attempt(ifo.AgentPass(), [{**LOOK, "tools": ["Read", "bash"]}])
    assert isinstance(refused, list) and "never" not in refused[0]
    assert "'tools' lists the tools a new step uses" in refused[0]


def test_map_runs_take_their_neighbour_s_timeout():
    sb = sb_mode.Solver()
    out = _attempt(sb, [{"op": "insert", "after": "frame", "id": "plan", "kind": "agent",
                         "text": "Plan.", "to": "solve"}])
    parent = out
    parent.state["graph"]["nodes"]["plan"]["timeout_s"] = 321
    parent = Candidate.from_state(parent.state)
    nodes = _nodes(_attempt(sb, [{"op": "map", "after": "plan", "over": "solver_prompt",
                                  "text": "Do one.", "to": "solve"}], parent=parent))
    runs = [n for k, n in nodes.items() if k.startswith("plan_map_") and k[-1].isdigit()]
    assert runs and all(n.get("timeout_s") == 321 for n in runs), runs


def test_the_route_line_names_every_operator_that_takes_a_route():
    line = core_ops.route_field_line({"model": ["m"]})
    line = line[line.index("("):line.index(")")]
    for name, entry in core_ops.OPERATORS.items():
        assert (f"`{name}`" in line) == (core_ops.ROUTE_FIELD in entry["fields"]), name


def test_the_loop_line_says_what_a_judge_without_tools_is_handed():
    line = core_ops.OPERATORS["loop"]["line"]
    assert "as text (a binary or very long file only named)" in line
    stripped, fields = core_ops.without_tools(line, core_ops.OPERATORS["loop"]["fields"])
    assert "'tools'" not in stripped and "tools" not in fields
