"""The request's "what each step carries, and which operators apply to it" section.

Each row is worked out by applying the core operators with their default arguments to the
request's own state, so what it says applies must apply, what it says is refused must be
refused with those words, and the alternative a refusal names must itself apply. These
tests hold the section to that on the S4 start state (livemath/skill_graph) and on a nested
fork, and pin the loop operator's line to what a loop actually builds.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, Optimizer, Proposal, base
from refactory_dsh.optim.memory import state_view

MODE = "livemath/skill_graph"


def _start_state() -> dict:
    _package, mode = registry.resolve(Path("."), MODE)
    return mode().state_dict()


def _check(state: dict, protected=()):
    """An attempt's verdict on the real state, as the loop's `edit_check` gives it."""
    parent = Candidate.from_state(copy.deepcopy(state))
    return lambda edit: Optimizer().attempt(Proposal(edit=edit), parent, (), False, (),
                                            protected=tuple(protected), record=False)


def _section(state: dict, protected=(), check=None, **kwargs) -> list[str]:
    view = state_view(state)
    return base.operator_fit_section(nn_edits.catalogue("broad"), view.node_states, view.edges,
                                     view.knobs, check or _check(state, protected), **kwargs)


def _rows(lines: list[str]) -> dict[str, list[str]]:
    table = next(line for line in lines if line.startswith("| step |"))
    rows = {}
    for line in table.splitlines()[2:]:
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        rows[cells[0]] = cells[1:]
    return rows


def _outcome(edit: dict, state: dict, protected: tuple = ()) -> str | None:
    """An attempt's verdict on one edit (`Optimizer.attempt`), as the section's reason
    text: None when it is accepted."""
    verdict = Optimizer().attempt(Proposal(edit=edit), Candidate.from_state(copy.deepcopy(state)),
                                  (), False, (), protected=protected, record=False)
    if not isinstance(verdict, list):
        return None
    return "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)


def _edits_by_label(state: dict, step: str) -> dict[str, dict]:
    from refactory_dsh.nn.graph import Graph

    graph = Graph.from_state(state["graph"])
    return dict(base._fit_edits(graph.nodes[step], graph, set(nn_edits.CORE_OPS)))


def _check_rows_are_true(state: dict, protected: tuple = ()) -> dict[str, list[str]]:
    """Every row's verdicts agree with an attempt's: an applying label is accepted, a
    refused one is refused with the reason the row gives (its replacing hint checked)."""
    rows = _rows(_section(state, protected=protected))
    assert rows
    for step, (_carries, applies, refused) in rows.items():
        edits = _edits_by_label(state, step)
        for label in filter(None, applies.split(", ")) if applies != "-" else ():
            assert _outcome(edits[label], state, protected) is None, (step, label)
        for entry in refused.split(" | ") if refused != "-" else ():
            labels, reason = entry.split(": ", 1)
            for label in labels.split(", "):
                got = _outcome(edits[label], state, protected)
                assert got is not None, (step, label)
                assert got == reason, (step, label, got)
    return rows


def test_each_row_on_the_s4_start_state_says_what_an_attempt_does():
    rows = _check_rows_are_true(_start_state())
    # With the mode's own protected steps (the data pass): merging or replacing its entry
    # passes the graph's checks but changes the protected data node, so an attempt refuses.
    _package, mode = registry.resolve(Path("."), MODE)
    guarded = _check_rows_are_true(_start_state(), protected=tuple(mode().edit_protected()))
    assert "merge" in rows["prepare"][1] and "merge" not in guarded["prepare"][1]
    assert "E-protected: node 'data'" in guarded["prepare"][2]
    model_steps = {"compare", "decide", "hypotheses", "precision"}
    assert model_steps <= set(rows)
    assert "fact" not in " ".join(rows)  # rows are steps, not facts
    assert rows["step_fork"][1] == "branch, serialize"


def test_a_model_step_s_row_follows_its_prompt_back_through_the_plain_op_that_built_it():
    rows = _rows(_section(_start_state()))
    assert rows["decide"][0].startswith(
        "decide_prompt (op decide_prompt, which reads compare_notes.txt (compare), "
        "hypotheses_notes.txt (hypotheses), precision_notes.txt (precision)")
    assert rows["precision"][0].startswith(
        "precision_prompt (op notes_prompts, which reads compare_notes.txt (compare)")
    assert rows["notes_prompts"][0] == "-"  # only model steps' prompts are traced
    # The text a plain op builds a prompt from is often a knob: named as one.
    # One op makes two prompts: its knobs are said to be the op's, not each prompt's.
    assert "precision_prompt (op notes_prompts, which reads compare_notes.txt (compare), " \
        "knob(s) precision_task, hypotheses_task (the op's, for all 2 of its outputs))" \
        in rows["precision"][0], rows["precision"][0]
    assert "decide_prompt (op decide_prompt, which reads" in rows["decide"][0]
    assert "knob decide_task)" in rows["decide"][0]  # one output: the knob is its own


def test_lineage_follows_plain_ops_back_to_their_sources():
    state = sf.Applier(sf.seeds()["fuzz_dataflow"]).seed_state
    carries = _rows(_section(state))["summarise"][0]
    # cleaned <- op clean <- raw.txt <- op load (which reads nothing): two ops back.
    clean = ("op clean, which reads raw.txt (op load, which reads nothing), knob(s) depth "
             "(the op's, for all 2 of its outputs)")
    assert carries == (f"cleaned ({clean}); source (op load, which reads nothing); "
                       f"clean.txt ({clean})"), carries


def test_lineage_stops_at_its_depth_and_says_so(monkeypatch):
    monkeypatch.setattr(base, "CARRIES_DEPTH", 1)
    state = sf.Applier(sf.seeds()["fuzz_dataflow"]).seed_state
    carries = _rows(_section(state))["summarise"][0]
    assert "raw.txt (op load, not followed further)" in carries, carries
    _package, mode = registry.resolve(Path("."), "livemath/solver")
    assert "knob(s) skill (the op's, for all 3 of its outputs)" in _rows(
        _section(mode().state_dict()))["solve"][0]


def test_a_refusal_names_the_exact_replacing_edit_and_that_edit_applies():
    state = _start_state()
    refused = _rows(_section(state))["decide"][2]
    hint = re.search(r"replace it with (\{.*?\}\]\})", refused)
    assert hint, refused
    edit = json.loads(hint.group(1))
    assert edit == {"op": "split", "step": "decide",
                    "into": [{"kind": "agent", "text": "<what it does>", "sees": ["task"]}]}
    assert _outcome(edit, state) is None
    # The same exact edit for a data pass's ends, where the replacement is code.
    reason = _outcome({"op": "remove", "step": "prepare"}, state)
    edit = json.loads(re.search(r"replace it with (\{.*?\}\]\})", reason).group(1))
    assert edit == {"op": "split", "step": "prepare", "into": [{"kind": "code"}]}
    assert _outcome(edit, state) is None


def test_a_replacing_hint_is_held_to_the_attempt_s_checks_not_only_the_graph_s():
    """The operators' precheck knows the graph only: for a protected step it would name a
    split an attempt refuses (E-protected). The attempt checks the hinted edit itself."""
    state = _start_state()
    rewire = {"op": "rewire", "step": "decide", "sees": ["task"]}
    assert '"op": "split"' in nn_edits.precheck(rewire, state)
    reason = _outcome(rewire, state, protected=("decide",))
    assert "replacing it is refused too (E-protected" in reason, reason
    assert '"op": "split"' not in reason
    # A code replacement refused the same way leaves no note about its default code.
    reason = _outcome({"op": "remove", "step": "prepare"}, state, protected=("data",))
    assert "replacing it is refused too (E-protected: node 'data'" in reason, reason
    assert "default code" not in reason and '"op": "split"' not in reason


def test_a_refusal_never_names_a_replacing_edit_that_would_be_refused():
    """Where the replacement is refused too (a fork's branch; a step alone reading a frozen
    knob), the hint says so with that refusal instead of naming the edit."""
    state = _start_state()
    refused = _rows(_section(state))["hypotheses"][2]
    assert "replacing it is refused too ('hypotheses' is a fork's branch" in refused, refused
    assert '"op": "split"' not in refused
    _package, mode = registry.resolve(Path("."), "livemath/solver")
    solver = mode().state_dict()
    reason = _outcome({"op": "remove", "step": "solve"}, solver)
    assert "replacing it is refused too ('solve' alone reads the frozen knob(s)" in reason
    for row in _rows(_section(solver)).values():
        for found in re.findall(r"replace it with (\{.*?\}\]\})", row[2]):
            assert _outcome(json.loads(found), solver) is None, found


def test_a_step_alone_reading_a_frozen_knob_is_shown_unsplittable():
    state = _start_state()
    assert "split" in _rows(_section(state))["compare"][1].split(", ")
    state["knobs"]["compare__system_prompt"]["spec"]["frozen"] = True
    rows = _check_rows_are_true(state)
    applies, refused = rows["compare"][1].split(", "), rows["compare"][2]
    assert "split" not in applies and "frozen knob(s) ['compare__system_prompt']" in refused
    assert "frozen knob" in _section(state)[0]


def test_a_step_alone_reading_a_frozen_knob_merges_only_with_a_prompt_naming_it():
    """The lead's rule, held to the attempt: merge, like split, is accepted when the new
    step's prompt names the frozen knob, and refused otherwise."""
    state = sf.Applier(sf.seeds()["fuzz_dataflow"]).seed_state
    state["knobs"]["depth"]["spec"]["frozen"] = True
    named = {"op": "merge", "steps": ["clean", "summarise"],
             "text": "Summarise {{source}} to depth {{depth}}"}
    unnamed = {**named, "text": "Summarise {{source}}"}
    assert _outcome(named, state) is None
    assert "frozen knob(s) ['depth']" in _outcome(unnamed, state)
    lead = _section(state)[0]
    assert "is not removed, and is split or merged only when the new step's prompt names" \
        in lead


def test_each_row_on_a_nested_fork_says_what_the_operators_do():
    state = sf.Applier(sf.seeds()["fuzz_nested_fork"]).seed_state
    rows = _check_rows_are_true(state)
    forks = [n for n, node in state["graph"]["nodes"].items() if node.get("kind") == "fork"]
    assert len(forks) >= 2 and set(forks) <= set(rows)


def test_a_protected_step_s_row_is_what_an_attempt_says_not_what_the_graph_allows():
    """Precheck and expansion alone accept a split of a protected step; an attempt refuses
    it (E-protected), and so does the row."""
    state = _start_state()
    assert nn_edits.precheck({"op": "split", "step": "decide"}, state) is None
    rows = _check_rows_are_true(state, protected=("decide",))
    applies, refused = rows["decide"][1].split(", "), rows["decide"][2]
    assert "split" not in applies and "E-protected" in refused, rows["decide"]
    # A protected neighbour too: merging decide_prompt into decide changes decide.
    assert "merge" in _rows(_section(state))["decide_prompt"][1]
    assert "merge" not in rows["decide_prompt"][1].split(", ")
    assert "merge: E-protected" in rows["decide_prompt"][2], rows["decide_prompt"]


def test_the_loop_s_request_takes_its_verdicts_from_the_run_s_own_attempt(tmp_path):
    """In a run the table is `Run.mechanics` on the edited candidate: an edit the run
    refuses (here: every edit, for a stub) is shown refused with the run's words."""
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig
    from refactory_dsh.outerloop import Run

    loop = strategies.create("default", MODE, LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path,
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    loop.population.append(seed)
    ctx = loop._context()
    run = Run(_loop=loop)
    edits = _edits_by_label(seed.state, "compare")
    for label, edit in edits.items():
        assert (ctx.edit_check(edit) == run.mechanics(Proposal(edit=edit), seed,
                                                      record=False)), label
    # The run's own protected steps reach the table: the data pass's entry merges in the
    # graph's terms but not in the run's.
    from test_structure_first_request import _section as section_of

    body = section_of(ctx.render(), "What each step carries, and which operators apply to it")
    rows = _rows([body[body.index("| step |"):]])
    assert "merge" not in rows["prepare"][1].split(", ")
    assert "merge: E-protected: node 'data'" in rows["prepare"][2], rows["prepare"]
    stub = base.operator_fit_section(ctx.catalogue, ctx.node_states, ctx.edges, ctx.knobs,
                                     lambda edit: ["E-run: refused by this run"])
    assert all(row[1] == "-" and row[2].endswith("E-run: refused by this run")
               for row in _rows(stub).values())


def test_the_section_stays_within_its_limit_and_names_what_it_left_out():
    state = _start_state()
    full = _section(state)
    assert sum(map(len, full)) <= base.OPERATOR_FIT_CHARS
    small = _section(state, limit=2500)
    assert sum(map(len, small)) <= 2500
    steps = len(_rows(full))
    shown = _rows(small)
    # The note counts the steps the table lists, not every node of the graph.
    assert small[2] == (f"({steps - len(shown)} of {steps} steps not listed, to stay within "
                        "2500 characters)"), small[2]
    # The rows kept are the shortened ones: each refusal cut to its first clause.
    for row in shown.values():
        for entry in row[2].split(" | ") if row[2] != "-" else ():
            assert ";" not in entry.split(": ", 1)[1], entry


def test_no_section_when_the_run_permits_no_core_operator_or_gives_no_check():
    state = _start_state()
    view = state_view(state)
    assert base.operator_fit_section(nn_edits.catalogue("knob_only"), view.node_states,
                                     view.edges, view.knobs, _check(state)) == []
    assert base.operator_fit_section(nn_edits.catalogue("broad"), view.node_states,
                                     view.edges, view.knobs, None) == []


def test_the_loop_line_states_the_cap_and_verdict_the_loop_builds():
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    after = nn_edits.apply({"op": "loop", "step": "review", "judge": "agent"}, state)
    knob = next(k for k in after["knobs"] if k.endswith("__max_iterations"))
    spec = after["knobs"][knob]["spec"]
    line = core_ops.OPERATORS["loop"]["line"]
    assert knob == "review_check__max_iterations" and "<gate>__max_iterations" in line
    assert "<step>_check" in line
    assert (f"default {spec['default']}: up to {spec['default'] + 1} runs; "
            f"{spec['lo']} to {spec['hi']}") in line
    judge = after["graph"]["nodes"]["review_check_judge"]
    assert "proceed" in json.dumps(judge) and "reloop" in json.dumps(judge)
    assert "proceed or reloop" in line


def test_an_edit_whose_preconditions_pass_but_whose_expansion_fails_is_shown_refused(
        monkeypatch):
    """`precheck` checks preconditions only; the row goes by the full expansion."""
    state = _start_state()
    real = nn_edits.apply

    def apply(edit, state, *args, **kwargs):
        if edit.get("op") == "insert" and edit.get("after") == "compare":
            raise nn_edits.EditError("the new step would read nothing", code="E-wiring")
        return real(edit, state, *args, **kwargs)

    monkeypatch.setattr(nn_edits, "apply", apply)
    applies, refused = _rows(_section(state))["compare"][1:]
    assert "insert" not in applies.split(", ")
    assert "insert: E-wiring: the new step would read nothing" in refused


def test_a_replacing_hint_whose_edit_fails_to_expand_says_it_is_refused(monkeypatch):
    state = _start_state()
    real = nn_edits.apply

    def apply(edit, state, *args, **kwargs):
        if edit.get("op") == "split" and edit.get("step") == "prepare":
            raise nn_edits.EditError("E-wiring: the code step would read nothing")
        return real(edit, state, *args, **kwargs)

    monkeypatch.setattr(nn_edits, "apply", apply)
    reason = nn_edits.precheck({"op": "remove", "step": "prepare"}, state)
    assert "replacing it is refused too (E-wiring: the code step would read nothing)" in reason


def test_the_request_shows_the_section_after_the_operators_in_domain_free_words():
    from refactory_dsh.optim import ProposalContext
    from test_proposal_presentation import DENYLIST
    from test_structure_first_request import _headings
    from test_structure_first_request import _section as section_of

    view = state_view(_start_state())

    def render(preset: str) -> str:
        return ProposalContext(mode_name="m", nodes=view.nodes, edges=view.edges,
                               node_states=view.node_states, knobs=view.knobs,
                               catalogue=nn_edits.catalogue(preset),
                               edit_check=_check(_start_state())).render()

    text = render("broad")
    headings = _headings(text)
    fit = "What each step carries, and which operators apply to it"
    assert headings.index("Structural operators") < headings.index(fit) < headings.index(
        "Each step's settings")
    body = section_of(text, fit)
    assert "| decide |" in body and "| step_fork |" in body
    lead = body.split("\n|", 1)[0]
    assert not [word for word in DENYLIST if word.lower() in lead.lower()]
    assert fit not in render("knob_only")


def test_a_code_replacing_hint_is_checked_without_a_code_dry_run_and_says_so(monkeypatch):
    """With recorded inputs a code edit is dry-run in a sandbox, a process per run; the
    hint's own check skips that (it would run on every request) and the hint says so."""
    from refactory_dsh.optim import checks

    given = []
    real = checks.code_dry_run

    def spy(parent_id, parent_state, state, nodes, recorded, sandbox=None):
        given.append(recorded)
        return real(parent_id, parent_state, state, nodes, None, sandbox)

    monkeypatch.setattr(checks, "code_dry_run", spy)
    optimizer = Optimizer()
    optimizer.recorded_inputs = lambda candidate: {"run": "x", "passes": {}}
    state = _start_state()
    verdict = optimizer.attempt(Proposal(edit={"op": "remove", "step": "prepare"}),
                                Candidate.from_state(state), (), False, (), record=False)
    reason = verdict[0]
    assert '"into": [{"kind": "code"}]} (not dry-run) (its default code' in reason, reason
    assert given and all(recorded is None for recorded in given), given


def test_the_lead_names_what_sees_task_gives_on_this_pipeline():
    """A branch that `sees: ["task"]` also gets compare's prompt fact (built by a plain op
    from the item and compare's instruction knob): the lead lists it, so a proposal does
    not describe such a step as seeing the question alone."""
    from refactory_dsh.nn import core_ops
    from refactory_dsh.nn.graph import Graph

    state = _start_state()
    lead = _section(state)[0]
    names = sorted(name for _kind, name in core_ops._task_inputs(
        Graph.from_state(state["graph"]), state))
    assert '`"sees": ["task"]` gives a new step these' in lead
    for name in ("compare_prompt", "user_prompt", "current_item.json"):
        assert name in names and name in lead.split('["task"]', 1)[1], name
    branch = nn_edits.apply({"op": "branch", "fork": "step_fork", "kind": "agent",
                             "text": "{{user_prompt}}", "sees": ["task"]}, state)
    # The branch that sees "task" does take compare's prompt fact in, as the lead says.
    from refactory_dsh.nn.validate import consumed

    graph = Graph.from_state(branch["graph"])
    agent = next(n for n in set(graph.nodes) - set(state["graph"]["nodes"])
                 if graph.nodes[n].kind == "agent")
    values = {k: e["value"] for k, e in branch["knobs"].items()}
    assert ("fact", "compare_prompt") in consumed(graph.nodes[agent], values)


def test_the_loop_line_says_what_the_judge_sees_and_how_often_the_step_runs():
    line = core_ops.OPERATORS["loop"]["line"]
    assert "The judge sees what the step read and wrote, unless 'sees' says otherwise" in line
    default = core_ops.LOOP_ITERATIONS[0]
    assert f"default {default}: up to {default + 1} runs" in line


def test_the_run_s_table_skips_the_code_dry_run_and_marks_new_code(tmp_path, monkeypatch):
    """On a graph with a code step and recorded inputs, an attempt dry-runs new or changed
    code in a sandbox, a process each time; the table asks for every operator on every
    request, so it skips that and marks an accepted edit that changes code."""
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig
    from refactory_dsh.optim import checks
    from test_structure_first_request import _section as section_of

    state = nn_edits.apply({"op": "insert", "after": "compare", "kind": "code", "id": "tally"},
                           _start_state())
    loop = strategies.create("default", MODE, LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path,
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(state)
    loop.population.append(seed)
    loop._editing_parent = seed.id
    loop.optimizer.recorded_inputs = lambda candidate: {"run": str(tmp_path), "passes": {}}
    given = []
    real = checks.code_dry_run

    def spy(parent_id, parent_state, state, nodes, recorded, sandbox=None):
        given.append(recorded)
        return real(parent_id, parent_state, state, nodes, None, sandbox)

    monkeypatch.setattr(checks, "code_dry_run", spy)
    body = section_of(loop._context().render(),
                      "What each step carries, and which operators apply to it")
    rows = _rows([body[body.index("| step |"):]])
    assert "merge (not dry-run)" in rows["tally"][1].split(", "), rows["tally"]
    assert "insert" in rows["tally"][1].split(", ")  # no code changed: no mark
    assert given and all(recorded is None for recorded in given), given


def test_a_loop_states_its_cap_in_the_edit():
    """`max_iterations` on the loop sets how often the judge may send the step back, so
    a proposal says its cap in the edit and not only in prose; outside 1-5 it is refused."""
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    one = nn_edits.apply({"op": "loop", "step": "review", "judge": "agent",
                          "max_iterations": 1}, state)
    assert one["knobs"]["review_check__max_iterations"]["value"] == 1
    assert one["graph"]["nodes"]["review_check"]["max_iterations"] == 1
    plain = nn_edits.apply({"op": "loop", "step": "review", "judge": "agent"}, state)
    assert plain["knobs"]["review_check__max_iterations"]["value"] == core_ops.LOOP_ITERATIONS[0]
    for bad in (0, 6, "2", True):
        reason = nn_edits.precheck({"op": "loop", "step": "review", "judge": "agent",
                                    "max_iterations": bad}, state)
        assert reason and "'max_iterations' is a whole number from 1 to 5" in reason, bad
    assert core_ops.OPERATORS["loop"]["example"]["max_iterations"] == core_ops.LOOP_ITERATIONS[0]


def test_a_fork_s_row_says_what_a_branch_seeing_the_task_takes_in():
    rows = _rows(_section(_start_state()))
    assert rows["step_fork"][0] == ("a new branch that sees the task takes in: compare_prompt "
                                    "(prepare), current_item.json (data), user_prompt (prepare)")
    nested = sf.Applier(sf.seeds()["fuzz_nested_fork"]).seed_state
    forks = [n for n, x in nested["graph"]["nodes"].items() if x.get("kind") == "fork"]
    assert all(_rows(_section(nested))[f][0].startswith("a new branch that sees the task")
               for f in forks)
