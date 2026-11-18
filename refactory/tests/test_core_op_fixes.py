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


# ── review: what 'takes_over' may rewire ───────────────────────────────────


def _fuzz(name: str) -> dict:
    return sf.Applier(sf.seeds()[name]).seed_state


def test_a_step_on_a_sibling_branch_cannot_be_taken_over():
    """right_a runs beside left, not before it: the new step would never see it."""
    reason = nn_edits.precheck({"op": "insert", "after": "left", "kind": "code", "id": "pick",
                                "takes_over": "right_a"}, _fuzz("fuzz_fork_join"))
    assert reason and "on every path to it" in reason


def test_a_branch_of_a_fork_whose_join_comes_first_can_be():
    state = _fuzz("fuzz_fork_join")
    after_join = {"op": "insert", "after": "merge", "kind": "code", "id": "pick",
                  "takes_over": "right_a"}
    assert nn_edits.precheck(after_join, state) is None


def test_a_knob_an_earlier_and_a_later_step_share_is_refused():
    """review reads {{draft}} through a knob finalize reads too: renamed, finalize would
    read the new step's output under the old name only if the knob kept it."""
    state = _fuzz("fuzz_draft_review")
    graph = Graph.from_state(state["graph"])
    shared = next((k for k in graph.nodes["review"].knobs if k in graph.nodes["finalize"].knobs),
                  None)
    if shared is None:
        shared = "shared_text"
        state = nn_edits.apply({"op": "add_knob", "name": shared,
                                "spec": {"default": "Check {{draft}}.", "kind": "str"}}, state)
        for node_id in ("review", "finalize"):
            node = state["graph"]["nodes"][node_id]
            node["knobs"] = [*node.get("knobs", []), shared]
    state["knobs"][shared]["value"] = "Check {{draft}}."
    reason = nn_edits.precheck({"op": "insert", "after": "review", "kind": "code", "id": "pick",
                                "takes_over": "draft"}, state)
    assert reason and "both read" in reason and repr(shared) in reason


def test_a_fact_a_reader_names_in_its_code_is_refused():
    state = nn_edits.apply({"op": "insert", "after": "draft", "kind": "code", "id": "count",
                            "sees": ["draft"],
                            "text": "def main(inp):\n    return {'count': len(inp['draft'])}\n"},
                           _fuzz("fuzz_draft_review"))
    reason = nn_edits.precheck({"op": "insert", "after": "review", "kind": "code", "id": "pick",
                                "takes_over": "draft"}, state)
    assert reason and "in its code or feedback keys" in reason and "'count'" in reason


def test_a_protected_reader_or_target_comes_back_e_protected():
    state = _fuzz("fuzz_draft_review")
    take = {"op": "insert", "after": "review", "kind": "code", "id": "pick", "takes_over": "draft"}
    with pytest.raises(nn_edits.EditError) as reader:
        nn_edits.apply(take, state, protected=("review",))
    assert reader.value.code == "E-protected" and "'review'" in str(reader.value)
    with pytest.raises(nn_edits.EditError) as target:
        nn_edits.apply(take, state, protected=("draft",))
    assert target.value.code == "E-protected" and "'draft'" in str(target.value)


def test_a_grading_reader_is_never_rewired():
    from refactory_dsh.optim import Candidate, Optimizer, Proposal

    state = _fuzz("fuzz_draft_review")
    seed = Candidate.from_state(state)
    refused = Optimizer().attempt(
        Proposal(edit={"op": "insert", "after": "review", "kind": "code", "id": "pick",
                       "takes_over": "draft"}),
        seed, (), True, (), graders=("review",))
    assert isinstance(refused, list) and refused[0].startswith("E-protected"), refused
    assert "'review' grades" in refused[0]


def test_an_agent_cannot_take_over_a_step_writing_two_files():
    reason = nn_edits.precheck({"op": "insert", "after": "plot", "kind": "agent", "text": "x",
                                "takes_over": "plot"}, _fuzz("fuzz_file_output"))
    graph = Graph.from_state(_fuzz("fuzz_file_output")["graph"])
    if len(graph.nodes["plot"].writes) < 2:
        pytest.skip("fuzz_file_output's plot writes one file")
    assert reason and "writes only its reply" in reason


def test_a_copy_has_exactly_the_knobs_of_the_step_it_copies():
    s4 = _mode("livemath/skill_graph").state_dict()
    nodes = nn_edits.apply({"op": "parallelize", "steps": ["decide"], "copies": 1},
                           s4)["graph"]["nodes"]
    own = [k.replace("decide__", "") for k in nodes["decide"]["knobs"]]
    copy = [k.replace("decide_copy1__", "") for k in nodes["decide_copy1"]["knobs"]]
    assert own == copy and "decide_copy1__task" not in nodes["decide_copy1"]["knobs"]


@pytest.mark.parametrize("tools,refused", [(["bash"], False), (False, False), (True, True),
                                           (["Bash"], True), ("yes", True)])
def test_tools_is_a_list_from_the_sealed_set_or_false(tools, refused):
    edit = {"op": "insert", "after": "solve", "kind": "agent", "text": "x", "tools": tools}
    reason = nn_edits.precheck(edit, _mode("livemath/solver").state_dict())
    assert bool(reason and "lists the tools a new step uses, from bash" in reason) == refused, reason


# ── finding 16: a bundle keeps the reader it gave a new step ────────────────


S4_BUNDLE = [{"op": "branch", "fork": "step_fork", "kind": "agent", "id": "independent",
              "text": "Solve it yourself.", "sees": ["task"]},
             {"op": "set_knob", "name": "independent__system_prompt", "value": "You are careful."}]


def _s4_attempt(*more: dict):
    from refactory_dsh.optim import Candidate, Optimizer, Proposal

    mode = _mode("livemath/skill_graph")
    seed = Candidate.from_state(mode.state_dict())
    return Optimizer().attempt(Proposal(edits=[*S4_BUNDLE, *more]), seed, mode.invariants(),
                               True, (), protected=mode.edit_protected())


def test_a_whole_value_that_drops_the_new_step_s_slot_is_refused():
    """S4-s43's kept 192548858db6: decide's prompt held independent's reply on 0 of 33."""
    refused = _s4_attempt({"op": "set_knob", "name": "decide_task", "value": "Decide now."})
    assert isinstance(refused, list) and refused[0].startswith("E-unread"), refused
    assert ("edit 3 (set_knob decide_task value) removes {{independent}}, the only read of "
            "the output of new step 'independent' (by decide_prompt)") in refused[0]


def _reaches(tmp_path, monkeypatch):
    patched = _s4_attempt({"op": "set_knob", "name": "decide_task",
                           "patch": [{"kind": "append", "content": "Weigh the independent answer."}]})
    assert not isinstance(patched, list), patched
    from refactory_packages.livemath.skill_graph import SkillGraph

    mode = SkillGraph()
    mode.load_state_dict(patched.state)
    return mode, patched.state


def test_with_a_patch_the_new_step_reaches_decide_on_the_walker(tmp_path, monkeypatch):
    mode, _state = _reaches(tmp_path, monkeypatch)
    model = _StepModel({"independent": "INDEPENDENT-REPLY"})
    _walk(mode, model, tmp_path, monkeypatch)
    decide = [prompt for node, prompt, _s in model.log if node == "decide"]
    assert decide and all("INDEPENDENT-REPLY" in prompt for prompt in decide)


@needs_spine
def test_with_a_patch_the_new_step_reaches_decide_on_the_spine(tmp_path, monkeypatch):
    from refactory_dsh import lowering

    mode, state = _reaches(tmp_path, monkeypatch)
    _patch_task(monkeypatch)
    # The candidate's knob values, as the DSH runtime hands them to its ops
    # (`dsh_runtime`: ``--knobs``); the test drain takes them from the environment.
    monkeypatch.setenv(lowering.KNOBS_ENV, lowering.knob_environment(state)[lowering.KNOBS_ENV])
    spine_dir = tmp_path / "spine"
    assert _spine(mode.graph(), spine_dir, tmp_path,
                  agent_responses={"independent": "INDEPENDENT-REPLY"}, mode_state=state) == 0
    for record in json.loads((spine_dir / "item_records.json").read_text()):
        decide = [e["prompt"] for e in record["exchanges"] if e["node"] == "decide"]
        assert decide and all("INDEPENDENT-REPLY" in p for p in decide), record["id"]


def test_as_built_says_a_new_step_no_step_reads():
    """The S4 bundle applied edit by edit (no attempt check): independent's output reaches
    no step, and the candidate's As built says so; with the slot kept, it does not."""
    from refactory_dsh.nn import pipeline_view

    parent = _mode("livemath/skill_graph").state_dict()
    state = parent
    for edit in [*S4_BUNDLE, {"op": "set_knob", "name": "decide_task", "value": "Decide now."}]:
        state = nn_edits.apply(edit, state)
    line = "! new step independent: its output reaches no step"
    assert line in pipeline_view.changed_lines(parent, state)
    kept = parent
    for edit in S4_BUNDLE:
        kept = nn_edits.apply(edit, kept)
    assert line not in pipeline_view.changed_lines(parent, kept)


def test_a_frozen_knob_naming_the_target_s_output_is_refused():
    state = _fuzz("fuzz_draft_review")
    knob = next(k for k in Graph.from_state(state["graph"]).nodes["review"].knobs
                if "{{draft}}" in str(state["knobs"][k]["value"]))
    state["knobs"][knob]["spec"] = {**state["knobs"][knob].get("spec", {}), "frozen": True}
    reason = nn_edits.precheck({"op": "insert", "after": "review", "kind": "code", "id": "pick",
                                "takes_over": "draft"}, state)
    assert reason and f"frozen {knob!r}" in reason


def test_a_new_step_ending_the_pass_leaves_its_output_to_the_task():
    """The choice step taking over solve ends the pass; the Task reads what it writes."""
    from refactory_dsh.nn import pipeline_view

    parent = _mode("livemath/solver").state_dict()
    _mode_, chosen = _solver(*CHOICE)
    assert not [line for line in pipeline_view.changed_lines(parent, chosen)
                if "reaches no step" in line]


# ── re-review: what a skip loses, and the readers a takeover renames ────────


ROUTE_SKIPPING_REVIEW = {"op": "route", "after": "draft", "judge": "agent", "text": "x",
                         "to": {"needs_review": "review", "fine": "finalize"}}


def test_a_route_skipping_a_step_a_later_prompt_reads_only_under_a_guard_is_taken():
    """finalize reads {{review}} only inside {% if review %}: a route that skips review
    loses nothing finalize needs (the validator's own rule for an optional slot)."""
    state = _fuzz("fuzz_draft_review")
    with pytest.raises(nn_edits.EditError, match="'finalize' would run without review"):
        nn_edits.apply(ROUTE_SKIPPING_REVIEW, state)
    guarded = nn_edits.apply({"op": "set_knob", "name": "finalize_task", "value": (
        "Write the final answer.\n\n## Draft\n{{draft}}"
        "{% if review %}\n\n## Review\n{{review}}{% endif %}")}, state)
    assert validate_state(nn_edits.apply(ROUTE_SKIPPING_REVIEW, guarded)) == []


def test_a_slot_a_knob_fills_is_never_lost():
    """A {{name}} a knob of that name fills is there on every path: no step need make it."""
    state = _fuzz("fuzz_draft_review")
    state = nn_edits.apply({"op": "add_knob", "name": "review",
                            "spec": {"default": "No review.", "kind": "str"}}, state)
    assert core_ops.lost_inputs(state, _skip_review(state)) is None


def _skip_review(state: dict) -> dict:
    """The state the route leaves, without the operator's own check."""
    steps = nn_edits._specs()["route"].expand(ROUTE_SKIPPING_REVIEW, dict(state))
    for step in steps:
        state = nn_edits.apply(step, state)
    return state


def test_a_pass_after_the_new_step_reads_the_new_step_s_output():
    """A takeover before a data node: the pass's steps run after the new one, so they
    read the new step's output under the old name, never the taken-over step's."""
    state = _fuzz("fuzz_data_pass")
    state = nn_edits.apply({"op": "add_node", "position": {"start": True}, "node": {
        "id": "guide", "kind": "agent", "prompt_template": "Write a hint for every item.",
        "writes": ["guide.txt"], "produces": ["guide"]}}, state)
    state["graph"]["nodes"]["solve"]["prompt_template"] = "Solve: {{question}}\nHint: {{guide}}"
    assert validate_state(state) == []
    after = nn_edits.apply({"op": "insert", "after": "guide", "kind": "code", "id": "pick",
                            "takes_over": "guide"}, state)
    graph = Graph.from_state(after["graph"])
    from refactory_dsh.nn.validate import consumed

    values = {k: e["value"] for k, e in after["knobs"].items()}
    assert ("fact", "guide") in consumed(graph.nodes["solve"], values)
    assert graph.nodes["pick"].produces == ("guide",)


def test_a_reader_whose_code_cannot_be_read_is_refused():
    """An op that takes the target's fact as an input reads it by name in code no
    renaming reaches: refused, whatever the op is."""
    state = _fuzz("fuzz_draft_review")
    old = state["graph"]["nodes"]["review"]
    state["graph"]["nodes"]["review"] = {
        "id": "review", "kind": "op", "call": "loop_ops:noop", "inputs": ["draft"],
        "writes": list(old.get("writes") or []), "produces": list(old.get("produces") or [])}
    state["knobs"] = {k: v for k, v in state["knobs"].items() if not k.startswith("review")}
    reason = nn_edits.precheck({"op": "insert", "after": "review", "kind": "code", "id": "pick",
                                "takes_over": "draft"}, state)
    assert reason and "'review' reads 'draft'" in reason and "in its code or feedback" in reason


def test_a_fact_a_reader_takes_as_feedback_is_refused():
    state = _fuzz("fuzz_draft_review")
    state["graph"]["nodes"]["review"]["feedback_keys"] = ["draft"]
    reason = nn_edits.precheck({"op": "insert", "after": "review", "kind": "code", "id": "pick",
                                "takes_over": "draft"}, state)
    assert reason and "'review' reads 'draft'" in reason


def test_a_copy_keeps_the_step_s_current_text_not_its_default():
    s4 = nn_edits.apply({"op": "set_knob", "name": "decide__system_prompt",
                         "value": "TUNED SYSTEM PROMPT"}, _mode("livemath/skill_graph").state_dict())
    assert s4["knobs"]["decide__system_prompt"]["spec"].get("default") != "TUNED SYSTEM PROMPT"
    after = nn_edits.apply({"op": "parallelize", "steps": ["decide"], "copies": 1}, s4)
    assert after["knobs"]["decide_copy1__system_prompt"]["value"] == "TUNED SYSTEM PROMPT"


def test_a_step_whose_inputs_cannot_be_read_is_said_not_skipped(monkeypatch):
    import importlib

    validate = importlib.import_module("refactory_dsh.nn.validate")
    state = _fuzz("fuzz_draft_review")
    after = _skip_review(state)

    def unreadable(node, values):
        raise ValueError("bad template")

    monkeypatch.setattr(validate, "template_facts", unreadable)
    said = core_ops.lost_inputs(state, after)
    assert said and "inputs cannot be read (ValueError: bad template)" in said


# ── a judge's reply is a verdict: default code cannot take it over ─────────


def test_a_judge_is_not_split_into_code_that_cannot_answer_its_verdict():
    """The fuzz found it: a loop's judge split into code steps with no text left the
    verdict gate reading what the default code joins, which never starts with a verdict, so
    every run failed. Refused, saying what to give; code with its own text is taken."""
    state = nn_edits.apply({"op": "loop", "step": "left", "judge": "agent"},
                           sf.Applier(sf.seeds()["fuzz_fork_join"]).seed_state)
    edit = {"op": "split", "step": "left_check_judge", "into": [{"kind": "code"}, {"kind": "code"}]}
    assert nn_edits.precheck(edit, state) == (
        "'left_check' reads its verdict from what 'left_check_judge' writes; a code step "
        "taking that over needs its own 'text' (code whose output starts with the verdict), "
        "or make it an agent")
    own = "def main(inp):\n    return {'files': {'left_check_judge.txt': 'proceed'}}\n"
    assert nn_edits.precheck({**edit, "into": [{"kind": "code"}, {"kind": "code", "text": own}]},
                             state) is None


# ── smoke dry run: an added step's output reaches a reader; 'reads' is never ignored ──


def _s4(*edits: dict):
    from refactory_dsh.optim import Candidate, Optimizer, Proposal

    mode = _mode("livemath/skill_graph")
    seed = Candidate.from_state(mode.state_dict())
    return Optimizer().attempt(Proposal(edits=list(edits)), seed, mode.invariants(), True, (),
                               protected=mode.edit_protected())


def test_an_insert_whose_output_nothing_reads_is_refused_naming_where_it_could_go():
    """S4: decide's prompt is built by decide_prompt from decide_task, so a step inserted
    before decide reaches nothing; the refusal names the insert that does, which applies."""
    refused = _s4({"op": "insert", "before": "decide", "kind": "agent", "text": "Check."})
    assert isinstance(refused, list) and refused[0].startswith("E-unread"), refused
    assert ("decide takes no knob: insert before decide_prompt with 'to': decide_task"
            in refused[0]), refused
    taken = _s4({"op": "insert", "before": "decide_prompt", "kind": "agent", "text": "Check.",
                 "to": "decide_task"})
    assert not isinstance(taken, list), taken
    # After compare, notes_prompts fills two prompts, so which one is the proposer's to say.
    refused = _s4({"op": "insert", "after": "compare", "kind": "agent", "text": "Check."})
    assert "in 'to': precision_task, hypotheses_task, decide_task" in refused[0], refused


def test_a_to_whose_reader_runs_before_the_new_step_is_refused():
    refused = _s4({"op": "insert", "before": "decide", "kind": "agent", "text": "Check.",
                   "to": "decide_task"})
    assert isinstance(refused, list) and refused[0].startswith("E-wiring"), refused
    assert "decide_prompt, which runs before" in refused[0], refused


def test_a_reader_a_later_edit_of_the_same_proposal_adds_is_a_reader():
    """Edits are judged as a batch: an insert with no reader of its own, then a patch that
    puts its output into a prompt, is taken."""
    alone = {"op": "insert", "after": "compare", "kind": "agent", "id": "check", "text": "Check."}
    assert isinstance(_s4(alone), list)
    taken = _s4(alone, {"op": "set_knob", "name": "decide_task",
                        "patch": [{"kind": "append", "content": "\n\n{{check}}"}]})
    assert not isinstance(taken, list), taken


@pytest.mark.parametrize("op", ["fallback", "map"])
def test_the_other_ops_that_add_a_step_are_held_to_the_same_rule(op):
    """A fallback writes its step's outputs, so what read them reads it; a map's combine is
    placed as insert's step is, and refused the same way when nothing takes it."""
    if op == "fallback":
        out = _s4({"op": "fallback", "step": "compare", "kind": "agent", "text": "Simpler."})
        assert not isinstance(out, list), out
    else:
        out = _s4({"op": "map", "after": "decide", "over": "decide", "text": "One."})
        assert isinstance(out, list) and out[0].startswith("E-unread"), out


@pytest.mark.parametrize("mode_id,after", [("livemath/solver", "solve"),
                                           ("livemath/skill_graph", "compare")])
@pytest.mark.parametrize("kind", ["agent", "code"])
def test_reads_on_an_agent_or_code_step_is_refused_pointing_at_sees(mode_id, after, kind):
    state = _mode(mode_id).state_dict()
    for edit in ({"op": "insert", "after": after, "kind": kind, "reads": ["answer.txt"]},
                 {"op": "fallback", "step": after, "kind": kind, "reads": ["answer.txt"]},
                 {"op": "map", "after": after, "over": after, "text": "One.",
                  "combine": {"kind": kind, "reads": ["answer.txt"]}}):
        reason = nn_edits.precheck(edit, state)
        assert reason and "gets a file by naming it in 'sees'" in reason, (edit, reason)
    # An op step's reads are its own (`_step_nodes`), and stay taken.
    assert nn_edits.precheck({"op": "insert", "after": after, "kind": "op",
                              "call": core_ops.select_call(), "reads": []}, state) is None


SEES_FILE = {"op": "insert", "after": "compare", "kind": "agent", "id": "check",
             "text": "Check the notes.", "sees": ["compare_notes.txt"], "to": "decide_task"}


def _sees_file_mode():
    from refactory_packages.livemath.skill_graph import SkillGraph

    out = _s4(SEES_FILE)
    assert not isinstance(out, list), out
    mode = SkillGraph()
    mode.load_state_dict(out.state)
    return mode, out.state


def test_a_file_named_in_sees_reaches_the_step_and_its_output_decide_on_the_walker(
        tmp_path, monkeypatch):
    mode, _state = _sees_file_mode()
    model = _StepModel({"compare": "COMPARE-NOTES", "check": "CHECKED"})
    _walk(mode, model, tmp_path, monkeypatch)
    check = [prompt for node, prompt, _s in model.log if node == "check"]
    decide = [prompt for node, prompt, _s in model.log if node == "decide"]
    assert check and all("COMPARE-NOTES" in prompt for prompt in check)
    assert decide and all("CHECKED" in prompt for prompt in decide)


@needs_spine
def test_a_file_named_in_sees_reaches_the_step_and_its_output_decide_on_the_spine(
        tmp_path, monkeypatch):
    from refactory_dsh import lowering

    mode, state = _sees_file_mode()
    _patch_task(monkeypatch)
    monkeypatch.setenv(lowering.KNOBS_ENV, lowering.knob_environment(state)[lowering.KNOBS_ENV])
    spine_dir = tmp_path / "spine"
    assert _spine(mode.graph(), spine_dir, tmp_path, mode_state=state,
                  agent_responses={"compare": "COMPARE-NOTES", "check": "CHECKED"}) == 0
    for record in json.loads((spine_dir / "item_records.json").read_text()):
        prompts = {e["node"]: e["prompt"] for e in record["exchanges"]}
        assert "COMPARE-NOTES" in prompts["check"] and "CHECKED" in prompts["decide"], record["id"]


def test_a_to_inside_a_parallel_or_route_step_is_honoured_never_ignored():
    """'to' is a step field: a step parallelize's 'new' or a route's 'to' adds goes where it
    says, and a 'to' whose reader runs before the step is refused, as insert's is."""
    applier = sf.Applier(sf.seeds()["livemath_skill_graph"])
    from refactory_dsh.optim import Candidate

    parent = Candidate.from_state(applier.seed_state)
    routed = applier.attempt(parent, {
        "op": "route", "after": "compare", "judge": "agent", "text": "Which?",
        "to": {"proceed": "notes_prompts",
               "other": {"kind": "agent", "id": "alt", "text": "Alt.", "to": "decide_task"}}},
        {parent.id})
    assert isinstance(routed, Candidate), routed
    assert "{{alt}}" in routed.state["knobs"]["decide_task"]["value"]
    assert "{{alt}}" not in routed.state["knobs"]["precision_task"]["value"]
    early = applier.attempt(parent, {
        "op": "route", "after": "decide_prompt", "judge": "agent", "text": "Which?",
        "to": {"proceed": "decide",
               "other": {"kind": "agent", "id": "alt", "text": "Alt.", "to": "decide_task"}}},
        {parent.id})
    assert isinstance(early, list) and early[0].startswith("E-wiring"), early
    # After compare, notes_prompts fills two prompts: only 'to' says which one takes each.
    beside = applier.attempt(parent, {"op": "parallelize", "after": "compare", "new": [
        {"kind": "agent", "id": "bg", "text": "BG.", "to": "hypotheses_task"},
        {"kind": "agent", "id": "ex", "text": "EX.", "to": "precision_task"}]}, {parent.id})
    assert isinstance(beside, Candidate), beside
    knobs = beside.state["knobs"]
    assert "{{bg}}" in knobs["hypotheses_task"]["value"] and "{{ex}}" in knobs["precision_task"]["value"]
    assert "{{ex}}" not in knobs["hypotheses_task"]["value"]
