"""Core-operator correctness, from the expressibility audit (audit/easy-dsl-gaps.md):

1. a choice among attempts reaches what is scored (``insert`` 'takes_over');
2. a ``parallelize`` copy is sent exactly what the step it copies is sent;
3. a field an operator would ignore is refused, never dropped;
5. an edit after which a step would run without an input it had is refused at edit time.

Each holds for any graph and on both runtimes where it runs."""

from __future__ import annotations

import json

import pytest

import structure_fuzz as sf
from refactory_dsh.nn import core_ops, validate_state
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.graph import Graph
from test_livemath_blocks import _facts, _patch_task, _spine, needs_spine
from test_livemath_skill_graph import _StepModel
from test_pipeline_view import _mode


def _solver(*edits: dict) -> tuple[object, dict]:
    mode = _mode("livemath/solver")
    state = mode.state_dict()
    for edit in edits:
        state = nn_edits.apply(edit, state)
    assert validate_state(state) == [], validate_state(state)
    mode.load_state_dict(state)
    return mode, state


def _walk(mode, model, tmp_path, monkeypatch, name="walker"):
    _patch_task(monkeypatch)
    workdir = tmp_path / name
    workdir.mkdir(parents=True, exist_ok=True)
    result = mode.run(workdir, model=model)
    assert result.exit_ok, result.error
    return result


# ── 1. a choice reaches what is scored ─────────────────────────────────────


CHOICE = [
    {"op": "parallelize", "steps": ["solve"], "copies": 2},
    {"op": "insert", "after": "solve_join", "kind": "agent", "id": "solve_choose",
     "text": "Name the best attempt: solve, solve_copy1 or solve_copy2."},
    {"op": "insert", "after": "solve_choose", "kind": "op", "call": core_ops.select_call(),
     "takes_over": "solve"},
]
ATTEMPTS = {"solve": "<answer>A</answer>", "solve_copy1": "<answer>C</answer>",
            "solve_copy2": "<answer>D</answer>"}


@pytest.mark.parametrize("pick,label", [("solve_copy1", "C"), ("solve", "A")])
def test_the_chosen_attempt_is_what_is_graded_on_the_walker(pick, label, tmp_path, monkeypatch):
    mode, state = _solver(*CHOICE)
    nodes = state["graph"]["nodes"]
    assert nodes["solve_choose_next"]["writes"] == ["answer.txt"]
    assert nodes["solve"]["writes"] == ["solve.txt"]
    result = _walk(mode, _StepModel({**ATTEMPTS, "solve_choose": pick}), tmp_path, monkeypatch)
    assert {r["prediction"] for r in result.item_records} == {label}


@needs_spine
def test_the_chosen_attempt_is_what_is_graded_on_the_spine(tmp_path, monkeypatch):
    mode, state = _solver(*CHOICE)
    _patch_task(monkeypatch)
    spine_dir = tmp_path / "spine"
    rc = _spine(mode.graph(), spine_dir, tmp_path,
                agent_responses={**ATTEMPTS, "solve_choose": "solve_copy1"}, mode_state=state)
    assert rc == 0
    records = json.loads((spine_dir / "item_records.json").read_text())
    assert records and {r["prediction"] for r in records} == {"C"}
    assert _facts(spine_dir)["item_scores"]


def test_taking_over_renames_the_step_s_outputs_for_the_steps_before_only():
    """Any graph: the step taken over writes under new names, the steps before the new one
    read them there, and the steps after it read the new step's under the old names."""
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    after = nn_edits.apply({"op": "insert", "after": "review", "kind": "code",
                            "id": "pick", "takes_over": "draft"}, state)
    assert validate_state(after) == []
    graph = Graph.from_state(after["graph"])
    assert graph.nodes["pick"].writes == ("draft.txt",) and graph.nodes["pick"].produces == ("draft",)
    assert graph.nodes["draft"].writes != ("draft.txt",)
    from refactory_dsh.nn.validate import consumed

    values = {k: e["value"] for k, e in after["knobs"].items()}
    review_takes = {n for _k, n in consumed(graph.nodes["review"], values)}
    finalize_takes = {n for _k, n in consumed(graph.nodes["finalize"], values)}
    assert "draft" not in review_takes and graph.nodes["draft"].produces[0] in review_takes
    assert "draft" in finalize_takes


def test_taking_over_a_step_that_runs_after_is_refused():
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    reason = nn_edits.precheck({"op": "insert", "after": "draft", "kind": "code",
                                "takes_over": "review"}, state)
    assert reason and "does not run before the new step" in reason


# ── 2. a copy is sent what the step is sent ────────────────────────────────


def test_a_copy_is_sent_exactly_what_the_step_is_sent_on_the_walker(tmp_path, monkeypatch):
    mode, state = _solver({"op": "parallelize", "steps": ["solve"], "copies": 1})
    knobs = state["knobs"]
    assert knobs["solve_copy1__system_prompt"]["value"] == knobs["solve__system_prompt"]["value"]
    assert knobs["solve_copy1__system_prompt"]["spec"].get("frozen") \
        == knobs["solve__system_prompt"]["spec"].get("frozen")
    model = _StepModel({})
    _walk(mode, model, tmp_path, monkeypatch)
    sent = {}
    for node, prompt, system in model.log:
        sent.setdefault(node, []).append((prompt, system))
    assert sent["solve"] and sent["solve_copy1"] == sent["solve"]


@needs_spine
def test_a_copy_is_sent_exactly_what_the_step_is_sent_on_the_spine(tmp_path, monkeypatch):
    mode, state = _solver({"op": "parallelize", "steps": ["solve"], "copies": 1})
    _patch_task(monkeypatch)
    spine_dir = tmp_path / "spine"
    assert _spine(mode.graph(), spine_dir, tmp_path, agent_responses={}, mode_state=state) == 0
    lowered = json.loads(next((tmp_path / f"graphs-{spine_dir.name}").glob("*.graph.json"))
                         .read_text())["nodes"]
    assert lowered["solve_copy1"]["system_prompt"] == lowered["solve"]["system_prompt"]
    for record in json.loads((spine_dir / "item_records.json").read_text()):
        prompts = {}
        for exchange in record["exchanges"]:
            prompts.setdefault(exchange["node"], []).append(exchange["prompt"])
        assert prompts["solve_copy1"] == prompts["solve"], record["id"]


def test_a_copy_takes_no_sees():
    reason = nn_edits.precheck({"op": "parallelize", "steps": ["solve"], "copies": 1,
                                "sees": ["task"]}, _mode("livemath/solver").state_dict())
    assert reason and "a copy is sent exactly what the step is sent" in reason


# ── 3. an ignored field is refused ─────────────────────────────────────────


@pytest.mark.parametrize("edit,said", [
    ({"op": "insert", "after": "solve", "text": "x", "role": "judge"}, "'role'"),
    ({"op": "insert", "after": "solve", "text": "x", "timeout_s": 60, "retry": {}},
     "'retry', 'timeout_s'"),
    ({"op": "insert", "after": "solve", "text": "x", "tools": ["Bash"]}, "'tools' is true"),
    ({"op": "split", "step": "solve", "into": [{"kind": "agent", "text": "a", "role": "x"}]},
     "a new step does not take 'role'"),
])
def test_a_field_an_operator_would_ignore_is_refused(edit, said):
    reason = nn_edits.precheck(edit, _mode("livemath/solver").state_dict())
    assert reason and said in reason and "would be ignored" in reason, reason


def test_system_is_offered_where_insert_takes_it():
    """'system' worked on insert but was not in its fields: it is now listed, from the
    same source the precondition checks."""
    assert "system" in core_ops.OPERATORS["insert"]["fields"]
    assert nn_edits.precheck({"op": "insert", "after": "solve", "kind": "agent", "text": "x",
                              "system": "S"}, _mode("livemath/solver").state_dict()) is None


# ── 5. no edit makes a step run without an input it had ────────────────────


def test_a_route_skipping_the_steps_a_later_step_reads_is_refused():
    """The audit's route_skip_existing: on the easy path decide_prompt would run with
    neither fork step's notes and fail every item that takes it."""
    s4 = _mode("livemath/skill_graph").state_dict()
    with pytest.raises(nn_edits.EditError) as refused:
        nn_edits.apply({"op": "route", "after": "compare", "judge": "agent", "text": "x",
                        "to": {"hard": "notes_prompts", "easy": "decide_prompt"}}, s4)
    assert "'decide_prompt' would run without" in str(refused.value)
    assert "precision_notes.txt (made by precision)" in str(refused.value)


def test_routes_that_keep_every_input_and_removals_are_still_taken():
    s4 = _mode("livemath/skill_graph").state_dict()
    assert nn_edits.apply({"op": "route", "after": "compare", "judge": "agent", "text": "x",
                           "to": {"a": "notes_prompts",
                                  "b": {"kind": "agent", "text": "y"}}}, s4)
    assert nn_edits.apply({"op": "remove", "step": "precision"}, s4)


def test_the_lost_input_rule_holds_on_any_graph():
    """A skip on a graph with no LiveMath in it: a route from intake straight to finalize
    leaves finalize without what draft and review make."""
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    graph = Graph.from_state(state["graph"])
    assert {"intake", "draft", "review", "finalize"} <= set(graph.nodes)
    with pytest.raises(nn_edits.EditError, match="would run without"):
        nn_edits.apply({"op": "route", "after": "intake", "judge": "agent", "text": "x",
                        "to": {"long": "draft", "short": "finalize"}}, state)
