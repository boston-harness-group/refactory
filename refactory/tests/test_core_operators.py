"""The core structural operators (`nn.core_ops`): one line each, the data wired by the system.

- **What each operator wires**, on the synthetic seeds (`structure_fuzz_graphs`): new steps'
  inputs and where their outputs go, fork targets and join sources, verdict edges, the
  loop's limit knob, what a removed step leaves behind.
- **Both runtimes**: each operator's result runs to the same facts and item scores on the
  walker and the spine (`structure_fuzz.check_state(spine=True)`), and a data pass at
  parallelism 1 and 3.
- **Mutation checks**: each operator's wiring switched off in turn, and its named check
  fails.
- **The request** (target 5 of the plan): one line and one example per core operator, and
  every composition example parses, uses only core operators and applies as written on
  every seed it is rendered for.
- **Programmatic use**: every operator applies with no text at all; the same operator on
  the same parent makes the same state; each costs about as many characters as a prompt
  patch.

Nothing here is about what a step is for: the seeds are a tool step in a workspace, a step
that writes a file that is not text, draft -> review -> finalize, a fork and its join, a
gate loop and a data pass.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

import structure_fuzz as sf
import structure_fuzz_graphs as fg
from refactory_dsh import snippet
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.graph import Graph
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.optim import base as base_mod
from refactory_dsh.optim.edit_count import count
from test_livemath_blocks import needs_spine

CODE = snippet.available()[0]
needs_code = pytest.mark.skipif(not CODE, reason="no code-node sandbox on this machine")


def _apply(seed: str, edit: Any, state: dict | None = None) -> tuple[sf.Applier, Candidate | list[str]]:
    applier = sf.Applier(sf.seeds()[seed])
    parent = Candidate.from_state(state or applier.seed_state)
    return applier, applier.attempt(parent, edit, {parent.id})


def _ok(seed: str, edit: Any, state: dict | None = None) -> tuple[sf.Applier, Candidate]:
    applier, out = _apply(seed, edit, state)
    assert isinstance(out, Candidate), (seed, edit, out)
    return applier, out


def _graph(candidate: Candidate) -> Graph:
    return Graph.from_state(candidate.state["graph"])


def _knob(candidate: Candidate, name: str) -> str:
    return str(candidate.state["knobs"][name]["value"])


def _head(graph: Graph, step: str) -> str:
    """The first node of a new step: its collect step when it has one, else itself."""
    gather = f"{step}_inputs"
    return gather if gather in graph.nodes else step


def _sound(applier: sf.Applier, candidate: Candidate, *, spine: bool = False) -> None:
    assert sf.check_state(applier, candidate.state, walker=True, spine=spine) == []


def _parallel(state: dict, n: int) -> dict:
    """``state`` with its data pass running ``n`` items at once."""
    return nn_edits.configure(state, "data_parallelism", n)


# ── insert ────────────────────────────────────────────────────────────────


def test_insert_wires_what_the_step_sees_and_where_its_output_goes():
    """After draft: the new agent sees draft's input and output as facts, writes its own
    file and fact, and its fact reaches review's prompt knob; one change, one line."""
    edit = {"op": "insert", "after": "draft", "kind": "agent", "text": "List the weak points."}
    applier, out = _ok("fuzz_draft_review", edit)
    graph = _graph(out)
    new = graph.nodes["draft_next"]
    assert [e.target for e in graph.out_edges("draft")] == ["draft_next"]
    assert [e.target for e in graph.out_edges("draft_next")] == ["review"]
    task = _knob(out, "draft_next__task")
    assert task.startswith("List the weak points.") and "{{brief}}" in task and "{{draft}}" in task
    assert new.writes == ("draft_next.txt",) and new.produces == ("draft_next",)
    assert new.tool_policy == "none" and new.role == "solver"
    assert "{{draft_next}}" in _knob(out, "review_task")
    assert out.notes and "put {{draft_next}} into knob 'review_task'" in out.notes[0]
    assert count([edit], applier.seed_state)["logical"] == 1
    _sound(applier, out)


def test_insert_hands_a_file_to_a_model_step_through_a_collect_step():
    """A step whose output is a file that is not text: the new agent gets it through a
    collect step (path and size, never a crash), its other inputs as facts."""
    applier, out = _ok("fuzz_file_output", {"op": "insert", "after": "plot", "kind": "agent",
                                            "text": "Say what the figure shows."})
    graph = _graph(out)
    gather = graph.nodes["plot_next_inputs"]
    assert gather.call == "refactory_dsh.nn.operators:collect"
    assert set(gather.reads) >= {"figure.png", "plot.txt"}
    assert "{{plot_next_inputs}}" in _knob(out, "plot_next__task")
    assert "{{plotted}}" in _knob(out, "plot_next__task")
    _sound(applier, out)


def test_insert_before_a_branch_head_takes_its_place_in_the_fork():
    applier, out = _ok("fuzz_fork_join", {"op": "insert", "before": "left", "kind": "agent",
                                          "text": "Prepare."})
    graph = _graph(out)
    fork = graph.nodes["split"]
    # left reads a file, so its new neighbour gets it through a collect step, which heads
    # the branch now.
    assert fork.targets == ("left_prep_inputs", "right_a")
    assert [e.target for e in graph.out_edges("left_prep_inputs")] == ["left_prep"]
    assert [e.target for e in graph.out_edges("left_prep")] == ["left"]
    _sound(applier, out)


@needs_spine
@pytest.mark.parametrize("edit", [
    {"op": "insert", "after": "draft", "kind": "agent", "text": "List the weak points."},
    {"op": "insert", "before": "finalize", "kind": "code"},
    {"op": "insert", "after": "intake"},
], ids=["agent after", "code before", "all defaults"])
def test_insert_runs_alike_on_the_walker_and_the_spine(edit):
    if edit.get("kind") == "code" and not CODE:
        pytest.skip("no code-node sandbox")
    applier, out = _ok("fuzz_draft_review", edit)
    _sound(applier, out, spine=True)


# ── remove ────────────────────────────────────────────────────────────────


def test_remove_takes_an_inserted_step_out_and_undoes_its_wiring():
    applier, inserted = _ok("fuzz_draft_review", {"op": "insert", "after": "draft",
                                                  "kind": "agent", "text": "Weak points."})
    out = applier.attempt(inserted, {"op": "remove", "step": "draft_next"}, {inserted.id})
    assert isinstance(out, Candidate), out
    assert out.id == Candidate.from_state(applier.seed_state).id, "back to the seed"


def test_remove_refuses_with_the_exact_reader():
    _applier, out = _apply("fuzz_draft_review", {"op": "remove", "step": "review"})
    assert isinstance(out, list) and "'finalize' needs review" in out[0]


def test_remove_a_gate_keeps_the_verdicts_path_and_drops_the_rest():
    applier, out = _ok("fuzz_multi_terminal", {"op": "remove", "step": "route", "keep": "proceed"})
    graph = _graph(out)
    assert {"route", "escalate", "reject"}.isdisjoint(graph.nodes)
    assert [e.target for e in graph.out_edges("classify")] == ["accept"]
    _sound(applier, out)


def test_remove_a_loop_gate_takes_its_judge_with_it():
    applier, looped = _ok("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": "agent"})
    out = applier.attempt(looped, {"op": "remove", "step": "draft_check", "keep": "proceed"},
                          {looped.id})
    assert isinstance(out, Candidate), out
    assert out.id == Candidate.from_state(applier.seed_state).id


def test_remove_a_branch_folds_a_fork_left_with_one():
    applier, branched = _ok("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"],
                                                  "copies": 1})
    out = applier.attempt(branched, {"op": "remove", "step": "draft_copy1"}, {branched.id})
    assert isinstance(out, Candidate), out
    assert out.id == Candidate.from_state(applier.seed_state).id


@needs_spine
@pytest.mark.parametrize("seed, edit", [
    ("fuzz_nested_fork", {"op": "remove", "step": "solo"}),
    ("fuzz_gate_loop", {"op": "remove", "step": "review", "keep": "proceed"}),
    ("fuzz_multi_terminal", {"op": "remove", "step": "route", "keep": "proceed"}),
], ids=["a branch", "a loop gate", "a route gate"])
def test_remove_runs_alike_on_the_walker_and_the_spine(seed, edit):
    applier, out = _ok(seed, edit)
    _sound(applier, out, spine=True)


# ── parallelize and serialize ─────────────────────────────────────────────


def test_parallelize_copies_run_beside_the_step_and_reach_the_next_reader():
    applier, out = _ok("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 2})
    graph = _graph(out)
    fork, join = graph.nodes["draft_fork"], graph.nodes["draft_join"]
    assert fork.targets == ("draft", "draft_copy1", "draft_copy2")
    assert set(join.sources) == {"draft", "draft_copy1", "draft_copy2"}
    assert _knob(out, "draft_copy1__task") == _knob(out, "draft_task")
    review = _knob(out, "review_task")
    assert "{{draft_copy1}}" in review and "{{draft_copy2}}" in review and "{{draft}}" in review
    _sound(applier, out)


def test_parallelize_then_serialize_is_a_round_trip_of_the_structure():
    applier, out = _ok("fuzz_dataflow", {"op": "parallelize", "after": "clean", "new": [
        {"kind": "agent", "text": "One angle."}, {"kind": "agent", "text": "Another angle."}]})
    back = applier.attempt(out, {"op": "serialize", "fork": "clean_fork"}, {out.id})
    assert isinstance(back, Candidate), back
    graph = _graph(back)
    assert "clean_fork" not in graph.nodes and "clean_join" not in graph.nodes
    # Each branch runs after the one before it (a branch that sees a file has its collect
    # step in front).
    assert [e.target for e in graph.out_edges("clean")] == [_head(graph, "clean_branch1")]
    assert [e.target for e in graph.out_edges("clean_branch1")] == [_head(graph, "clean_branch2")]
    _sound(applier, back)


def test_parallelize_refuses_steps_that_read_each_other():
    _a, out = _apply("fuzz_draft_review", {"op": "parallelize", "steps": ["review", "finalize"]})
    assert isinstance(out, list) and "'finalize' reads ['review']" in out[0]


def test_serialize_a_seed_fork_runs_its_branches_in_order():
    applier, out = _ok("fuzz_fork_join", {"op": "serialize", "fork": "split"})
    graph = _graph(out)
    assert [e.target for e in graph.out_edges("setup")] == ["left"]
    assert [e.target for e in graph.out_edges("left")] == ["right_a"]
    assert [e.target for e in graph.out_edges("right_b")] == ["decide"]
    _sound(applier, out)


@needs_spine
@pytest.mark.parametrize("seed, edit", [
    ("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 2}),
    ("fuzz_dataflow", {"op": "parallelize", "after": "clean", "new": [
        {"kind": "agent", "text": "One angle."}, {"kind": "agent", "text": "Another."}]}),
    ("fuzz_fork_join", {"op": "serialize", "fork": "split"}),
    ("fuzz_nested_fork", {"op": "serialize", "fork": "inner"}),
], ids=["copies", "new steps", "serialize", "serialize nested"])
def test_parallelize_and_serialize_run_alike_on_the_walker_and_the_spine(seed, edit):
    applier, out = _ok(seed, edit)
    _sound(applier, out, spine=True)


# ── branch ────────────────────────────────────────────────────────────────


def test_branch_adds_a_step_to_the_fork_and_its_output_after_the_join():
    applier, out = _ok("fuzz_fork_join", {"op": "branch", "fork": "split", "kind": "agent",
                                          "text": "A third view.", "sees": ["task"]})
    graph = _graph(out)
    assert graph.nodes["split"].targets == ("left", "right_a", _head(graph, "split_branch"))
    assert set(graph.nodes["merge"].sources) == {"left", "right_b", "split_branch"}
    assert [e.target for e in graph.out_edges("split_branch")] == ["merge"]
    _sound(applier, out)


def test_an_optional_branch_is_marked_optional_and_read_only_when_there():
    applier, out = _ok("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 1})
    more = applier.attempt(out, {"op": "branch", "fork": "draft_fork", "optional": True,
                                 "kind": "agent", "text": "Try a bolder draft."}, {out.id})
    assert isinstance(more, Candidate), more
    graph = _graph(more)
    knobs = {name: entry["value"] for name, entry in more.state["knobs"].items()}
    assert graph.optional_targets_of(graph.nodes["draft_fork"], knobs) == (
        _head(graph, "draft_fork_branch"),)
    assert "{% if draft_fork_branch %}" in _knob(more, "review_task")
    _sound(applier, more)


@needs_spine
@needs_code
def test_an_optional_branch_that_fails_leaves_the_item_running_on_both_runtimes():
    applier, out = _ok("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 1})
    failing = applier.attempt(out, {"op": "branch", "fork": "draft_fork", "optional": True,
                                    "kind": "code", "id": "flaky",
                                    "text": "def main(inp):\n    raise ValueError('no')\n"},
                              {out.id})
    assert isinstance(failing, Candidate), failing
    assert sf.check_state(applier, failing.state, walker=True, spine=True) == []


@needs_spine
@pytest.mark.parametrize("seed, edit", [
    ("fuzz_fork_join", {"op": "branch", "fork": "split", "kind": "agent", "text": "Third."}),
    ("fuzz_nested_fork", {"op": "branch", "fork": "inner", "kind": "agent", "text": "Deeper."}),
], ids=["a fork", "a nested fork"])
def test_branch_runs_alike_on_the_walker_and_the_spine(seed, edit):
    applier, out = _ok(seed, edit)
    _sound(applier, out, spine=True)


# ── rewire ────────────────────────────────────────────────────────────────


def test_rewire_to_the_task_makes_a_step_independent():
    """review sees only the task: its draft slot is taken out of its prompt, the brief
    stays, and it keeps its id, its outputs and its prompt knob."""
    applier, out = _ok("fuzz_draft_review", {"op": "rewire", "step": "review", "sees": ["task"]})
    task = _knob(out, "review_task")
    assert "{{draft}}" not in task and "{{brief}}" in task
    review = _graph(out).nodes["review"]
    assert review.writes == ("review.txt",) and ("prompt_template", "review_task") in review.field_bindings
    _sound(applier, out)


def test_rewire_adds_what_it_is_told_to_see():
    applier, out = _ok("fuzz_draft_review", {"op": "rewire", "step": "finalize",
                                             "sees": ["task", "review"]})
    task = _knob(out, "finalize_task")
    assert "{{review}}" in task and "{{brief}}" in task and "{{draft}}" not in task
    _sound(applier, out)


@needs_spine
def test_rewire_runs_alike_on_the_walker_and_the_spine():
    applier, out = _ok("fuzz_draft_review", {"op": "rewire", "step": "review", "sees": ["task"]})
    _sound(applier, out, spine=True)


# ── loop ──────────────────────────────────────────────────────────────────


def test_loop_with_an_agent_judge_reloops_through_a_verdict_gate_with_a_limit_knob():
    applier, out = _ok("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": "agent",
                                             "text": "Is the draft complete?"})
    graph = _graph(out)
    gate, judge = graph.nodes["draft_check"], graph.nodes["draft_check_judge"]
    assert gate.evaluator == snippet.VERDICT and gate.required_reads == ("draft_check_judge.txt",)
    assert ("max_iterations", "draft_check__max_iterations") in gate.field_bindings
    assert out.state["knobs"]["draft_check__max_iterations"]["value"] == 2
    assert {(e.target, e.condition) for e in graph.out_edges("draft_check")} == {
        ("draft", "reloop"), ("review", "proceed")}
    assert "proceed or reloop" in _knob(out, "draft_check_judge__task")
    assert judge.gate_kind == "fn" or judge.kind == "agent"
    _sound(applier, out)


@needs_spine
@pytest.mark.parametrize("judge", ["agent", "code", fg.PROCEED])
def test_loop_with_each_judge_kind_runs_alike_on_the_walker_and_the_spine(judge):
    if judge == "code" and not CODE:
        pytest.skip("no code-node sandbox")
    applier, out = _ok("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": judge})
    _sound(applier, out, spine=True)


def test_loop_never_makes_a_gate_a_person_or_an_agent_answers():
    for judge in ("agent", "code", fg.PROCEED):
        if judge == "code" and not CODE:
            continue
        _a, out = _ok("fuzz_draft_review", {"op": "loop", "step": "draft", "judge": judge})
        assert all(n.gate_kind == "fn" for n in _graph(out).nodes.values() if n.kind == "gate")


# ── route ─────────────────────────────────────────────────────────────────


ROUTE = {"op": "route", "after": "solve", "judge": "agent", "text": "Which way?",
         "to": {"proceed": "check", "again": {"kind": "agent", "text": "Try again."},
                "simplify": {"kind": "agent", "text": "Simplify it."}}}


def test_route_chooses_among_labelled_steps_that_rejoin():
    applier, out = _ok("fuzz_data_pass", ROUTE)
    graph = _graph(out)
    labels = {(e.target, e.condition) for e in graph.out_edges("solve_route")}
    assert labels == {("check", "proceed"), (_head(graph, "solve_again"), "again"),
                      (_head(graph, "solve_simplify"), "simplify")}
    assert [e.target for e in graph.out_edges("solve_again")] == ["check"]
    assert {"solve_route", "solve_again", "solve_simplify"} <= graph.data_body("items")
    _sound(applier, out)


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_route_to_three_branches_inside_a_data_pass_runs_alike_on_both_runtimes(parallelism):
    applier = sf.Applier(sf.seeds()["fuzz_data_pass"])
    applier.seed_state = _parallel(applier.seed_state, parallelism)
    parent = Candidate.from_state(applier.seed_state)
    out = applier.attempt(parent, ROUTE, {parent.id})
    assert isinstance(out, Candidate), out
    _sound(applier, out, spine=True)


# ── split, replace, merge ─────────────────────────────────────────────────


def test_split_into_two_steps_the_second_taking_over_the_outputs():
    applier, out = _ok("fuzz_draft_review", {"op": "split", "step": "finalize", "into": [
        {"kind": "agent", "text": "Outline the answer."},
        {"kind": "agent", "text": "Write it out."}]})
    graph = _graph(out)
    assert "finalize" not in graph.nodes and "finalize_task" not in out.state["knobs"]
    assert graph.nodes["finalize_b"].writes == ("final.txt",)
    assert "{{finalize_a}}" in _knob(out, "finalize_b__task")
    _sound(applier, out)


def test_split_with_one_step_replaces_it():
    applier, out = _ok("fuzz_draft_review", {"op": "split", "step": "review", "into": [
        {"kind": "agent", "text": "Check the draft line by line."}]})
    graph = _graph(out)
    assert "review" not in graph.nodes
    assert graph.nodes["review_new"].produces == ("review",)
    assert [e.target for e in graph.out_edges("draft")] == ["review_new"]
    _sound(applier, out)


def test_merge_makes_two_steps_one_with_the_first_inputs_and_the_second_outputs():
    applier, out = _ok("fuzz_draft_review", {"op": "merge", "steps": ["review", "finalize"]})
    graph = _graph(out)
    merged = graph.nodes["review_finalize"]
    assert merged.writes == ("final.txt",) and merged.produces == ("final",)
    assert {"review", "finalize"}.isdisjoint(graph.nodes)
    assert not {"review_task", "finalize_task"} & set(out.state["knobs"])
    _sound(applier, out)


def test_merge_keeps_only_the_knobs_its_step_reads_and_says_which_went():
    """``depth`` was the op ``clean``'s setting; the merged agent's prompt never names it,
    so it goes, and the notes say so. The merged text names what ``clean`` made instead of
    calling it an input the step is not given."""
    applier, out = _ok("fuzz_dataflow", {"op": "merge", "steps": ["clean", "summarise"]})
    merged = _graph(out).nodes["clean_summarise"]
    assert "depth" not in merged.knobs and "depth" not in out.state["knobs"]
    text = _knob(out, "clean_summarise__task")
    assert "not given to this step" not in text
    assert "First do what step 'clean' did" in text and "after cleaned" in text
    notes = core_ops.wiring_notes({"op": "merge", "steps": ["clean", "summarise"]},
                                  applier.seed_state, out.state)
    assert notes and "dropped knob(s) depth, which no step reads now" in notes[0]
    _sound(applier, out)


def test_merge_keeps_a_shared_knob_its_text_names():
    _applier, out = _ok("fuzz_dataflow", {"op": "merge", "steps": ["clean", "summarise"],
                                          "text": "Clean {{source}} to depth {{depth}}."})
    assert "depth" in _graph(out).nodes["clean_summarise"].knobs


def test_merge_refuses_when_another_step_reads_what_the_first_makes():
    """``load`` makes raw.txt, which ``clean`` and also ``emit`` read: merging ``load`` into
    ``clean`` would leave ``emit`` reading a file nothing makes."""
    _applier, out = _apply("fuzz_dataflow", {"op": "merge", "steps": ["load", "clean"]})
    assert isinstance(out, list) and any("emit" in reason and "'load' makes" in reason
                                          for reason in out), out


def test_rewire_replaces_the_collect_step_it_made_before():
    """A second rewire of a step that already has its collect step replaces that step: one
    collect step, reading only what the step now sees."""
    applier, once = _ok("fuzz_dataflow", {"op": "rewire", "step": "summarise", "sees": ["clean"]})
    assert _graph(once).nodes["summarise_inputs"].reads == ("clean.txt",)
    _applier, twice = _ok("fuzz_dataflow", {"op": "rewire", "step": "summarise", "sees": ["load"]},
                          once.state)
    graph = _graph(twice)
    assert [n for n in graph.nodes if n.startswith("summarise_inputs")] == ["summarise_inputs"]
    assert graph.nodes["summarise_inputs"].reads == ("raw.txt",)
    notes = core_ops.wiring_notes({"op": "rewire", "step": "summarise", "sees": ["load"]},
                                  once.state, twice.state)
    assert notes and "took clean.txt, cleaned away from summarise" in notes[0]
    _sound(applier, twice)


def test_remove_says_what_it_took_from_the_steps_it_kept():
    """``solo`` wrote solo.txt, which ``end`` read: removing it takes that away from ``end``,
    and the notes name it."""
    applier, out = _ok("fuzz_nested_fork", {"op": "remove", "step": "solo"})
    notes = core_ops.wiring_notes({"op": "remove", "step": "solo"}, applier.seed_state, out.state)
    assert notes and "took solo.txt away from end" in notes[0]
    assert "solo__system_prompt" not in notes[0]


def test_a_removed_steps_own_knobs_are_not_news():
    """``review_task`` and ``finalize_task`` are the merged steps' own prompt knobs: they go
    with the steps, and the notes do not list them as dropped settings."""
    edit = {"op": "merge", "steps": ["review", "finalize"]}
    applier, out = _ok("fuzz_draft_review", edit)
    assert {"review_task", "finalize_task"} <= set(applier.seed_state["knobs"]) - set(out.state["knobs"])
    notes = core_ops.wiring_notes(edit, applier.seed_state, out.state)
    assert notes and "removed finalize, review" in notes[0] and "dropped" not in notes[0], notes


def test_splitting_a_step_that_uses_tools_says_its_tools_went():
    applier = sf.Applier(sf.seeds()["fuzz_draft_review"])
    state = json.loads(json.dumps(applier.seed_state))
    state["graph"]["nodes"]["review"].pop("tool_policy", None)  # the default: tools
    edit = {"op": "split", "step": "review", "into": [{"kind": "agent", "text": "Review {{draft}}."}]}
    _applier, out = _ok("fuzz_draft_review", edit, state)
    notes = core_ops.wiring_notes(edit, state, out.state)
    assert notes and "the tools of review went with it" in notes[0], notes

@needs_spine
@pytest.mark.parametrize("seed, edit", [
    ("fuzz_draft_review", {"op": "split", "step": "finalize"}),
    ("fuzz_draft_review", {"op": "split", "step": "review", "into": [{"kind": "agent",
                                                                      "text": "Check."}]}),
    ("fuzz_draft_review", {"op": "merge", "steps": ["review", "finalize"]}),
    ("fuzz_data_pass", {"op": "split", "step": "solve"}),
], ids=["split", "replace", "merge", "split in a pass"])
def test_split_replace_and_merge_run_alike_on_the_walker_and_the_spine(seed, edit):
    applier, out = _ok(seed, edit)
    _sound(applier, out, spine=True)


# ── copies of a step that uses tools ──────────────────────────────────────


def test_a_copy_of_a_tool_step_keeps_its_tools_in_a_fresh_workspace_of_its_own():
    applier, out = _ok("fuzz_tool_step", {"op": "parallelize", "steps": ["gather"], "copies": 1})
    copy = _graph(out).nodes["gather_copy1"]
    assert copy.tool_policy is None and copy.workspace == {"kind": "scratch", "collect": "files"}
    _sound(applier, out)


def test_a_new_tool_step_outside_a_fresh_workspace_is_still_refused():
    state = sf.Applier(sf.seeds()["fuzz_tool_step"]).seed_state
    after = nn_edits.apply({"op": "parallelize", "steps": ["gather"], "copies": 1}, state)
    after["graph"]["nodes"]["gather_copy1"].pop("workspace")
    assert base_mod._new_node_tools(state, after, False)


def test_a_tool_step_working_in_the_run_directory_is_not_copied_beside_itself():
    applier = sf.Applier(sf.seeds()["fuzz_tool_step"])
    state = json.loads(json.dumps(applier.seed_state))
    state["graph"]["nodes"]["gather"].pop("workspace")
    _a, out = _apply("fuzz_tool_step", {"op": "parallelize", "steps": ["gather"], "copies": 1},
                     state)
    assert isinstance(out, list) and "uses tools in the run directory" in out[0]


# ── programmatic use: defaults, determinism, size ─────────────────────────


#: Each operator with no text at all, on a seed it fits.
DEFAULTS = [
    ("fuzz_draft_review", {"op": "insert", "after": "draft"}),
    ("fuzz_draft_review", {"op": "remove", "step": "draft"}),
    ("fuzz_draft_review", {"op": "parallelize", "steps": ["draft"], "copies": 1}),
    ("fuzz_fork_join", {"op": "serialize", "fork": "split"}),
    ("fuzz_fork_join", {"op": "branch", "fork": "split"}),
    ("fuzz_draft_review", {"op": "rewire", "step": "review"}),
    ("fuzz_draft_review", {"op": "loop", "step": "draft"}),
    ("fuzz_draft_review", {"op": "route", "after": "review"}),
    ("fuzz_draft_review", {"op": "split", "step": "finalize"}),
    ("fuzz_draft_review", {"op": "merge", "steps": ["review", "finalize"]}),
    ("fuzz_draft_review", {"op": "map", "after": "draft", "over": "draft"}),
]


def test_every_core_operator_has_a_default_case():
    assert {edit["op"] for _seed, edit in DEFAULTS} == set(core_ops.CORE_OPS)


@pytest.mark.parametrize("seed, edit", DEFAULTS, ids=lambda v: v if isinstance(v, str) else v["op"])
def test_every_operator_applies_with_no_text_and_the_same_way_twice(seed, edit):
    if edit == {"op": "remove", "step": "draft"}:
        _a, out = _apply(seed, edit)
        assert isinstance(out, list), "removing a step another needs is refused, with the reader"
        return
    applier, first = _ok(seed, edit)
    _again, second = _ok(seed, edit)
    assert first.id == second.id
    _sound(applier, first)


@pytest.mark.parametrize("seed, edit", DEFAULTS, ids=lambda v: v if isinstance(v, str) else v["op"])
def test_an_operator_line_is_about_the_size_of_a_prompt_patch(seed, edit):
    """Target 1 of the plan: a structural edit at most about twice a prompt patch."""
    patch = {"op": "set_knob", "name": "review_task",
             "patch": [{"kind": "append", "content": "Name the weakest claim first."}]}
    assert len(json.dumps(edit)) <= 2 * len(json.dumps(patch))


# ── mutation checks: switch an operator's wiring off, and its check fails ─


def _breaks(monkeypatch, target: str, replacement: Any, seed: str, edit: dict | list,
            holds: Any) -> None:
    """With ``target`` replaced, the edit (or the last of a list, applied after the others)
    is refused or ``holds`` (its named check) fails."""
    edits = edit if isinstance(edit, list) else [edit]

    def run() -> tuple[sf.Applier, Candidate | list[str]]:
        applier = sf.Applier(sf.seeds()[seed])
        current = Candidate.from_state(applier.seed_state)
        for step in edits:
            outcome = applier.attempt(current, step, {current.id})
            if isinstance(outcome, list):
                return applier, outcome
            current = outcome
        return applier, current

    applier, out = run()
    assert isinstance(out, Candidate) and holds(out), "the check holds without the mutation"
    monkeypatch.setattr(core_ops, target, replacement)
    _a, mutated = run()
    assert isinstance(mutated, list) or not holds(mutated) or sf.check_state(
        applier, mutated.state, walker=True) != []


MUTATIONS = [
    ("_place", lambda *a, **k: [], "fuzz_draft_review",
     {"op": "insert", "after": "draft", "kind": "agent", "text": "Weak points."},
     lambda c: "{{draft_next}}" in _knob(c, "review_task")),
    ("_seen", lambda *a, **k: [], "fuzz_draft_review",
     {"op": "insert", "after": "draft", "kind": "agent", "text": "Weak points."},
     lambda c: "{{draft}}" in _knob(c, "draft_next__task")),
    ("_renamed", lambda *a, **k: [], "fuzz_fork_join",
     {"op": "insert", "before": "left", "kind": "agent", "text": "Prepare."},
     lambda c: "left_prep_inputs" in _graph(c).nodes["split"].targets),
    ("_fold_forks", lambda *a, **k: [], "fuzz_nested_fork",
     {"op": "remove", "step": "deep_a"},
     lambda c: "inner" not in _graph(c).nodes),
    ("_strip_steps", lambda *a, **k: [], "fuzz_draft_review",
     [{"op": "insert", "after": "draft", "kind": "agent", "text": "Weak points."},
      {"op": "remove", "step": "draft_next"}],
     lambda c: "{{draft_next}}" not in _knob(c, "review_task")),
    ("_strip_unseen", lambda text, keep: text, "fuzz_draft_review",
     {"op": "rewire", "step": "review", "sees": ["task"]},
     lambda c: "{{draft}}" not in _knob(c, "review_task")),
    ("_respec", lambda *a, **k: [], "fuzz_fork_join",
     {"op": "branch", "fork": "split", "kind": "agent", "text": "Third."},
     lambda c: _head(_graph(c), "split_branch") in _graph(c).nodes["split"].targets),
    ("_iterations_knob", lambda gate_id: (f"{gate_id}__max_iterations", {"kind": "int",
                                                                          "default": 9}),
     "fuzz_draft_review", {"op": "loop", "step": "draft", "judge": "agent"},
     lambda c: c.state["knobs"]["draft_check__max_iterations"]["value"] == 2),
    ("_drop_orphans", lambda build: None, "fuzz_draft_review",
     {"op": "merge", "steps": ["review", "finalize"]},
     lambda c: "finalize_task" not in c.state["knobs"]),
    ("_serialize_steps", lambda state, fork: [], "fuzz_fork_join",
     {"op": "serialize", "fork": "split"},
     lambda c: "split" not in _graph(c).nodes),
    ("_readers_from", lambda graph, node_id, state: [], "fuzz_draft_review",
     {"op": "parallelize", "steps": ["draft"], "copies": 1},
     lambda c: "{{draft_copy1}}" in _knob(c, "review_task")),
    ("_shared_knobs", lambda *nodes: [], "fuzz_dataflow",
     {"op": "merge", "steps": ["clean", "summarise"],
      "text": "Clean {{source}} to depth {{depth}}, then summarise it."},
     lambda c: "depth" in _graph(c).nodes["clean_summarise"].knobs),
    ("_carried_knobs", lambda kind, text, system, shared: list(shared), "fuzz_dataflow",
     {"op": "merge", "steps": ["clean", "summarise"]},
     lambda c: "depth" not in _graph(c).nodes["clean_summarise"].knobs
     and "depth" not in c.state["knobs"]),
]


@pytest.mark.parametrize("target, replacement, seed, edit, holds", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_an_operators_wiring_switched_off_is_caught(monkeypatch, target, replacement,
                                                             seed, edit, holds):
    _breaks(monkeypatch, target, replacement, seed, edit, holds)


# ── the request ───────────────────────────────────────────────────────────


def _context(seed: str, **extra: Any) -> base_mod.ProposalContext:
    state = sf.seeds()[seed].state()
    graph = state["graph"]
    return base_mod.ProposalContext(
        catalogue=nn_edits.catalogue("broad"), node_states=list(graph["nodes"].values()),
        nodes=[{"id": n, "kind": v["kind"]} for n, v in graph["nodes"].items()],
        edges=list(graph["edges"]),
        knobs=[{"name": k, "value": v["value"], "kind": v["spec"]["kind"], "frozen": False}
               for k, v in state["knobs"].items()], code_nodes=CODE, **extra)


def test_the_request_explains_each_core_operator_in_one_line_with_one_example():
    text = _context("fuzz_draft_review").render()
    section = text.split("## Structural operators", 1)[1].split("Compositions built", 1)[0]
    for name in core_ops.CORE_OPS:
        lines = [line for line in section.splitlines() if line.startswith(f"- `{name}` (")]
        assert len(lines) == 1, (name, lines)
        assert core_ops.OPERATORS[name]["line"] in lines[0]
        assert text.count(core_ops.OPERATORS[name]["line"]) == 1, "said once in the request"
        example = json.loads(lines[0].split(" E.g. ", 1)[1])
        assert example["op"] == name
    assert "none is the preferred shape; compose your own" in text


@pytest.mark.parametrize("seed", sorted(sf.seeds()))
def test_every_composition_example_uses_only_core_operators_and_applies_as_written(seed):
    ctx = _context(seed, new_node_tools=True)
    examples = base_mod.composition_examples(
        ctx.catalogue, ctx.node_states, ctx.edges, protected=list(sf.Applier(
            sf.seeds()[seed]).protected), code_nodes=CODE, tools=True,
        knob_values={k["name"]: k["value"] for k in ctx.knobs})
    if seed in ("fuzz_draft_review", "fuzz_dataflow"):
        titles = {title for title, _b in examples}
        assert {"Several attempts and a choice", "Check and send back",
                "Check a later step and send the work back to an earlier one",
                "Another attempt beside it, sent exactly what it is sent",
                "A background pass beside an experiment", "A tool step before reasoning"} <= titles
        # The code-step shapes are offered only where code steps can run (`snippet.available`).
        code_titles = {"A simulation step, then an analysis of its results",
                       "Iterative fitting: repeat a step until a code check passes"}
        assert (code_titles <= titles) if CODE else not code_titles & titles, titles
    for title, bundle in examples:
        assert all(edit["op"] in core_ops.CORE_OPS for edit in bundle), title
        json.loads(json.dumps(bundle))
        applier = sf.Applier(sf.seeds()[seed])
        applier.optimizer.allow_new_node_tools = True
        parent = Candidate.from_state(applier.seed_state)
        proposal = Proposal(edits=tuple(bundle), parent=parent.id)
        out = applier.optimizer.attempt(proposal, parent, applier.invariants, True, (),
                                        [parent.id], seed_state=applier.seed_state,
                                        protected=applier.protected, graders=applier.graders,
                                        allow_new_node_tools=True)
        assert isinstance(out, Candidate), (seed, title, out)


@pytest.mark.parametrize("seed", sorted(sf.seeds()))
def test_every_operator_example_that_names_this_graphs_steps_applies_as_written(seed):
    """An operator's example names this graph's own steps only where the operator applies to
    them (else it keeps its placeholders), so a proposer copying it gets a valid edit."""
    ctx = _context(seed)
    applier = sf.Applier(sf.seeds()[seed])
    lines = base_mod.operator_lines(ctx.catalogue, ctx.node_states, ctx.edges,
                                    protected=applier.protected,
                                    knob_values={k["name"]: k["value"] for k in ctx.knobs})
    named = 0
    for line in (line for line in lines if line.startswith("- `")):
        example = json.loads(line.split(" E.g. ", 1)[1])
        ids = [example.get(k) for k in ("step", "steps", "after", "before", "fork")]
        if "<" in json.dumps(ids) or "<" in json.dumps(list((example.get("to") or {}).values())):
            continue
        named += 1
        parent = Candidate.from_state(applier.seed_state)
        out = applier.attempt(parent, example, {parent.id})
        assert isinstance(out, Candidate), (seed, example, out)
    assert named >= (6 if seed == "fuzz_draft_review" else 1)


def test_the_compositions_rotate_and_name_this_graphs_steps():
    first = _context("fuzz_draft_review", generation=0).render()
    second = _context("fuzz_draft_review", generation=1).render()
    head = lambda text: text.split("compose your own:", 1)[1].split("\n- ", 2)[1]  # noqa: E731
    assert head(first) != head(second)
    assert '"steps": ["draft"]' in first


def test_the_answer_asks_a_structural_proposal_for_its_reasoning_and_shows_one_line():
    text = base_mod.answer_instructions(nn_edits.catalogue("broad"))
    assert '"addresses"' in text and '"flow_change"' in text
    assert '{"op": "insert", "after": "<existing step id>", "kind": "agent"' in text
    knob_only = base_mod.answer_instructions(nn_edits.catalogue("knob_only"))
    assert '"addresses"' not in knob_only and "## Structural operators" not in knob_only


def test_a_knob_only_run_sees_no_operator():
    assert base_mod.operator_lines(nn_edits.catalogue("knob_only"), [], []) == []


# ── reasoning fields and the prediction check ─────────────────────────────


def test_a_structural_proposal_missing_its_reasoning_is_marked_not_refused():
    edit = {"op": "insert", "after": "draft", "kind": "agent", "text": "x"}
    assert Proposal(edit=edit).missing_reasoning() == ["addresses", "flow_change", "prediction"]
    full = Proposal(edit=edit, addresses="draft fails on items a, b", flow_change="review sees "
                    "the weak points", prediction="items a and b fixed")
    assert full.missing_reasoning() == []
    assert Proposal(edit={"op": "set_knob", "name": "k", "value": 1}).missing_reasoning() == []


def test_the_answer_parser_keeps_addresses_and_flow_change():
    from refactory_dsh.optim.random_optimizer import PROPOSAL_FIELDS, _proposal_edits

    raw = {"op": "insert", "after": "draft", "kind": "agent", "text": "x",
           "addresses": "a", "flow_change": "b"}
    steps, _why = _proposal_edits(raw)
    assert steps == [{"op": "insert", "after": "draft", "kind": "agent", "text": "x"}]
    assert {"addresses", "flow_change"} <= set(PROPOSAL_FIELDS)


def test_the_prediction_check_says_what_happened_to_the_named_items_and_steps():
    parent = [{"id": "a", "passed": False, "score": 0.0,
               "exchanges": [{"node": "review", "response": "old"}]},
              {"id": "b", "passed": True, "score": 1.0, "exchanges": []}]
    child = [{"id": "a", "passed": True, "score": 1.0,
              "exchanges": [{"node": "review", "response": "new"},
                            {"node": "draft_next", "response": "weak points"}]},
             {"id": "b", "passed": True, "score": 1.0, "exchanges": []}]
    lines = base_mod.prediction_check("draft_next fixes a; review changes; b stays", parent,
                                      child, ["review", "draft_next", "draft"])
    assert lines[0] == "Named items: a fixed, b unchanged."
    assert "draft_next: new, replied on 1 of 2 item(s)" in lines[1]
    assert "review: reply changed on 1 of the 1 item(s) it ran on" in lines[1]
    assert base_mod.prediction_check("better overall", parent, child, ["review"]) == []


def test_what_the_last_edit_changed_shows_the_wiring_and_the_prediction_check():
    pair = {"child": "c1", "parent": "p", "edit": [{"op": "insert", "after": "draft"}],
            "child_records": [{"id": "a", "passed": True, "score": 1.0}],
            "parent_records": [{"id": "a", "passed": False, "score": 0.0}],
            "prediction": "item a is fixed", "notes": ["insert: added draft_next"],
            "steps": ["draft"]}
    text = "\n".join(base_mod._edit_change_lines([pair]))
    assert "Wired by the system: insert: added draft_next" in text
    assert "Prediction check -- Named items: a fixed." in text


# ── time and cost, in numbers ─────────────────────────────────────────────


def test_time_per_item_is_said_in_numbers():
    lines = base_mod.time_lines({"per_item_median_s": 120, "per_item_p90_s": 300,
                                 "per_item_max_s": 1500})
    text = " ".join(lines)
    assert "120 s per item at the median" in text
    # A session ends on inactivity: no step time limit is stated.
    assert "may run for up to" not in text and "limit" not in text
    assert "300 s at the 90th percentile, 1500 s the longest" in text
    # A timed-out item is said once, in "Item results": not here, and not as a target.
    assert "timed out" not in text and "as a timeout" not in text
    assert "wall time" not in text and "at once" not in text
    assert base_mod.time_lines({}) == []


def test_one_timeout_moves_the_longest_never_the_median():
    """S1-s42's figure was its wall time x 12 at once / 35 items: 628 s, set by one item that
    ran into the 1800 s limit. Each item's own time gives the same median with it or
    without it."""
    from refactory_dsh.loop import item_time_spread

    usual = [{"id": str(i), "duration_s": 100.0 + i} for i in range(34)]
    without = item_time_spread(usual)
    with_timeout = item_time_spread([*usual, {"id": "slow", "duration_s": 1831.0,
                                              "error_kind": "timeout"}])
    assert with_timeout["per_item_median_s"] - without["per_item_median_s"] <= 1
    assert with_timeout["per_item_max_s"] == 1831 and without["per_item_max_s"] == 133
    assert item_time_spread([{"id": "a"}]) == {}
    # The 90th percentile is the 9th of 10, not the longest.
    spread = item_time_spread([{"id": str(i), "duration_s": float(i)} for i in range(1, 11)])
    assert (spread["per_item_p90_s"], spread["per_item_max_s"]) == (9, 10)


def test_cost_is_plainly_no_reason_without_a_cost_term_and_shown_per_item_with_one():
    zero = _context("fuzz_draft_review", parsimony=0.0).render()
    assert "cost is not a reason for or against a change in this run." in zero
    assert "cost per item" not in zero
    row = {"id": "c1", "parent": None, "score": 0.5, "structural": False, "edits": [],
           "cost_per_item": 0.0123}
    priced = _context("fuzz_draft_review", parsimony=0.01, leaderboard=[row]).render()
    assert "cost per item" in priced and "0.0123" in priced
    assert "cost is not a reason for or against" not in priced


def test_the_time_facts_are_of_one_candidate_when_no_parent_was_selected():
    """With no parent selected, the records stand for the latest candidate's parent; its
    items' times are that candidate's too, not the leaderboard best's."""
    from types import SimpleNamespace

    from refactory_dsh.loop import Loop

    state = sf.seeds()["fuzz_dataflow"].state()
    bred_from = SimpleNamespace(id="p", parent=None, duration_s=40.0, state=state)
    best = SimpleNamespace(id="best", parent=None, duration_s=4000.0, state=state)
    child = SimpleNamespace(id="c", parent="p", duration_s=None, state=state)
    loop = SimpleNamespace(model=SimpleNamespace(timeout_s=600), strategy_decision={},
                           _editing_parent=None, population=[best, bred_from, child],
                           _train_item_records={"p": [{"id": "a", "duration_s": 10.0},
                                                      {"id": "b", "repeat": 1,
                                                       "duration_s": 12.0},
                                                      {"id": "b", "repeat": 2,
                                                       "error_kind": "timeout",
                                                       "duration_s": 600.0}],
                                                "best": [{"id": "x", "duration_s": 999.0}]})
    loop._train_records_owner = lambda board: Loop._train_records_owner(loop, board)
    facts = Loop._time_facts(loop, [best])
    # The timed-out run is not among the times: it is said once, in "Item results".
    assert (facts["per_item_median_s"], facts["per_item_max_s"]) == (11, 12)
    assert facts["timed_items"] == 2
    assert "per_item_s" not in facts


def test_a_harness_timeout_is_read_from_structured_fields_only():
    from refactory_dsh.optim.base import harness_timeout

    assert harness_timeout({"error_kind": "timeout"})
    assert harness_timeout({"missing_steps": {"check": {"cause": "timeout"}}})
    # The backstop is the harness's too; a wall the step declared is the pipeline's.
    assert harness_timeout({"error_kind": "harness_incident"})
    assert harness_timeout({"missing_steps": {"check": {"cause": "harness_incident"}}})
    assert not harness_timeout({"error_kind": "declared_timeout"})
    assert not harness_timeout({"missing_steps": {"check": {"cause": "declared_timeout"}}})
    assert not harness_timeout({"passed": True, "error_kind": "timeout"})
    # Text never decides: a reply quoted in an error, a grader's reason, a word.
    for text in ("solve: CompletionTimeout: model completion timed out after 600s",
                 "check: RuntimeError: the reply said 'timed out after 1800s'",
                 "fit: ValueError: the timeout knob must be > 0"):
        assert not harness_timeout({"error_text": text, "error_kind": "RuntimeError"}), text
    assert not harness_timeout({"fail_reason": "MCQ=0: the timeout clause is wrong"})


@pytest.mark.parametrize("error,kind", [
    # The harness ended a model call: by type, or by the runtimes' anchored text.
    ("completion", "timeout"),
    ("walker", "timeout"),
    ("spine", "timeout"),
    ("inactivity", "timeout"),
    # The pipeline's own: a quoted reply, a code step's subprocess limit, a snippet limit.
    ("quoted", "RuntimeError"),
    ("quoted prefix", "RuntimeError"),
    ("expired", "TimeoutExpired"),
    ("snippet", "SnippetTimeout"),
])
def test_an_item_error_is_a_timeout_only_when_the_harness_ended_a_call(error, kind):
    import subprocess

    from refactory_dsh.items import error_kind
    from refactory_dsh.model import CompletionTimeout
    from refactory_dsh.snippet import SnippetTimeout

    made = {
        "completion": lambda: CompletionTimeout("model completion timed out after 1800s"),
        "walker": lambda: RuntimeError("solve: CompletionTimeout: model completion timed out "
                                       "after 1800s; requested runtime shutdown"),
        "spine": lambda: RuntimeError("solve exited with code 1\nworker: model completion "
                                      "timed out after 1800s; requested runtime shutdown"),
        "inactivity": lambda: RuntimeError("solve: CompletionTimeout: model completion timed out "
                                           "after 600s of inactivity"),
        "quoted": lambda: RuntimeError("check: ValueError: the reply was 'model completion "
                                       "timed out after 1800s'"),
        "quoted prefix": lambda: RuntimeError('check: ValueError: got "x: CompletionTimeout: y"'),
        "expired": lambda: subprocess.TimeoutExpired(["pytest"], 10),
        "snippet": lambda: SnippetTimeout("code step ran past its 30 s limit"),
    }[error]
    assert error_kind(made()) == kind


# ── a macro on a state an earlier edit of a bundle left with an unreadable cycle ──


#: An edge that closes a cycle avoiding every gate, per seed: legal mid-bundle (only the
#: state a bundle reaches is validated), and every macro must refuse it, not crash.
GATELESS_CYCLE = {"fuzz_fork_join": {"op": "add_edge", "source": "merge", "target": "setup"},
                  "fuzz_nested_fork": {"op": "add_edge", "source": "inner_join",
                                       "target": "begin"},
                  "fuzz_data_pass": None}


def _gateless(seed: str) -> dict:
    state = sf.Applier(sf.seeds()[seed]).seed_state
    edge = GATELESS_CYCLE[seed]
    if edge is None:  # close the cycle inside the data pass's body
        graph = Graph.from_state(state["graph"])
        body = sorted(graph.data_body("items"))
        last = next(n for n in body if not graph.out_edges(n) or all(
            e.target not in body for e in graph.out_edges(n)))
        edge = {"op": "add_edge", "source": last, "target": graph.nodes["items"].subgraph_entry}
    return nn_edits.apply(edge, state)


def test_route_after_a_bundle_edit_that_closed_a_gateless_cycle_is_refused_by_name():
    """Nightly long tier, seed 1 (#674): a bundle's ``add_edge`` closed a cycle through a
    join that avoids every gate, and the ``route`` after it crashed reading the graph's
    retries (``E-internal: GraphError``). It is refused, naming the edit and the cycle."""
    bundle = {"edits": [GATELESS_CYCLE["fuzz_fork_join"],
                        {"op": "route", "after": "decide", "judge": "agent"}]}
    _applier, out = _apply("fuzz_fork_join", bundle)
    assert isinstance(out, list) and out, out
    assert "[edit 2 of 2:" in out[0] and "E-state" in out[0] and "avoids every gate" in out[0], out
    assert "E-internal" not in out[0]


@pytest.mark.parametrize("seed", sorted(GATELESS_CYCLE))
def test_no_macro_crashes_on_a_state_with_a_gateless_cycle(seed):
    """Every core operator, drawn many ways, on a graph with a cycle that avoids every gate:
    precheck answers and apply applies or refuses (`EditError`); none raises anything
    else. Nested forks, a fork/join and a data pass's body."""
    import random

    state = _gateless(seed)
    applied = 0
    for name in core_ops.CORE_OPS:
        for draw in range(20):
            edit = core_ops.sample(name, state, random.Random(draw))
            if nn_edits.precheck(edit, state):
                continue
            try:
                nn_edits.apply(edit, state)
            except nn_edits.EditError:
                continue
            applied += 1
    assert applied, "no operator applied anywhere: the draws test nothing"


def test_a_macro_that_breaks_a_readable_graph_still_raises(monkeypatch):
    """Only a cycle the edit started from is a refusal: a GraphError from a macro on a
    graph whose cycles read is the operator's own bug, left to surface (E-internal)."""
    import dataclasses

    from refactory_dsh.nn.graph import GraphError

    spec = nn_edits._specs()["route"]

    def broken(edit, state):
        raise GraphError("the operator's own bug")

    monkeypatch.setitem(nn_edits._MACROS, "route",
                        dataclasses.replace(spec, expand=broken))
    state = sf.Applier(sf.seeds()["fuzz_fork_join"]).seed_state
    with pytest.raises(GraphError, match="own bug"):
        nn_edits.apply({"op": "route", "after": "decide", "judge": "agent"}, state)
    monkeypatch.setitem(nn_edits._MACROS, "route",
                        dataclasses.replace(spec, precondition=broken))
    with pytest.raises(GraphError, match="own bug"):
        nn_edits.precheck({"op": "route", "after": "decide", "judge": "agent"}, state)
    assert "avoids every gate" in nn_edits.precheck(
        {"op": "route", "after": "decide", "judge": "agent"}, _gateless("fuzz_fork_join"))


def test_a_copied_code_step_returns_its_own_outputs_even_when_main_calls_itself():
    """A copy of a code step renames what the original returns to the copy's names; a
    ``main`` that calls itself reaches the wrapper and is renamed twice, harmlessly."""
    from refactory_dsh.nn.graph import Node

    code = ("def main(inp):\n"
            "    if inp.get('again'):\n"
            "        return main({})\n"
            "    return {'facts': {'x_fact': 1}, 'files': {'x.txt': 't'}, 'note': 'kept'}\n")
    source = Node("x", "op", call=snippet.RUN, writes=("x.txt",), produces=("x_fact",))
    wrapped = core_ops._renamed_code(code, source, "x_a", ["x_a"], ["x_a.txt"])
    assert snippet.static_issues(wrapped) == []
    space: dict[str, Any] = {}
    exec(wrapped, space)  # noqa: S102 - the test runs the code it built
    expected = {"facts": {"x_a": 1}, "files": {"x_a.txt": "t"}, "note": "kept"}
    assert space["main"]({}) == expected
    assert space["main"]({"again": True}) == expected
    dropped: dict[str, Any] = {}
    exec(core_ops._renamed_code(code, source, "x_b", ["x_b"], []), dropped)  # noqa: S102
    assert dropped["main"]({}) == {"facts": {"x_b": 1}, "files": {}, "note": "kept"}
    assert core_ops._renamed_code(code, source, "x", ["x_fact"], ["x.txt"]) == code

