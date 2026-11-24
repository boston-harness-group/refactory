"""A refusal's replacing hint and the loop operator's line say what an attempt does.

A refusal that points to replacing a step names the exact edit only when an attempt would
accept it (`optim.base.checked_hint`), and otherwise says why replacing it is refused too;
the loop operator's line states the cap and verdict the loop builds.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import structure_fuzz as sf
from refactory_dsh import registry
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, Optimizer, Proposal

MODE = "livemath/skill_graph"


def _start_state() -> dict:
    _package, mode = registry.resolve(Path("."), MODE)
    return mode().state_dict()


def _check(state: dict, protected=()):
    """An attempt's verdict on the real state, as the loop's `edit_check` gives it."""
    parent = Candidate.from_state(copy.deepcopy(state))
    return lambda edit: Optimizer().attempt(Proposal(edit=edit), parent, (), False, (),
                                            protected=tuple(protected), record=False)


def _outcome(edit: dict, state: dict, protected: tuple = (), step: str = "") -> str | None:
    """An attempt's verdict on one edit (`Optimizer.attempt`), as the section's reason
    text: None when it is accepted."""
    verdict = Optimizer().attempt(Proposal(edit=edit), Candidate.from_state(copy.deepcopy(state)),
                                  (), False, (), protected=protected, record=False)
    if not isinstance(verdict, list):
        return None
    return "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)


def test_a_replacing_hint_is_held_to_the_attempt_s_checks_not_only_the_graph_s():
    """The operators' precheck knows the graph only: for a protected step it would name a
    split an attempt refuses (E-protected). The attempt checks the hinted edit itself."""
    state = _start_state()
    rewire = {"op": "rewire", "step": "decide", "sees": ["task"]}
    assert '"op": "rewire"' in nn_edits.precheck(rewire, state).split(core_ops.OWN_TEXT_HINT)[1]
    reason = _outcome(rewire, state, protected=("decide",))
    assert "giving it its own text is refused too (E-protected" in reason, reason
    assert core_ops.OWN_TEXT_HINT not in reason
    # A code replacement refused the same way leaves no note about its default code.
    reason = _outcome({"op": "remove", "step": "prepare"}, state, protected=("data",))
    assert "replacing it is refused too (E-protected: node 'data'" in reason, reason
    assert "default code" not in reason and '"op": "split"' not in reason


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


def test_the_loop_line_says_what_the_judge_sees_and_how_often_the_step_runs():
    line = core_ops.OPERATORS["loop"]["line"]
    assert "The judge is handed what the step read and wrote, unless 'sees' says otherwise" in line
    default = core_ops.LOOP_ITERATIONS[0]
    assert f"default {default}: up to {default + 1} runs" in line


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


def test_max_iterations_is_refused_on_the_written_out_loop():
    """The written-out loop's cap is its gate's own field: a top-level one would be
    ignored there (even out of range), so it is refused; the one-line form takes it."""
    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    gate = {"id": "review_gate", "kind": "gate", "evaluator": "refactory_dsh.snippet:gate",
            "code": "def main(inp):\n    return {'verdict': 'proceed', 'reason': ''}\n",
            "max_iterations": 2}
    written = {"op": "loop", "target": "review", "gate": gate}
    assert nn_edits.precheck(written, state) is None
    for cap in (2, 9):
        reason = nn_edits.precheck({**written, "max_iterations": cap}, state)
        assert reason and "'max_iterations' goes with the one-line loop" in reason, cap
    # The help names it in the one-line form, the only form it describes.
    loop = next(op for op in nn_edits.catalogue("broad") if op["op"] == "loop")
    text = json.dumps(loop)
    assert "'max_iterations'" in text and "written out" not in text, text



def test_an_unreadable_graph_gives_placeholder_examples_and_warns(caplog):
    from refactory_dsh.optim import base

    with caplog.at_level("WARNING"):
        assert base._example_state([{"id": "a", "kind": "no-such-kind"}], []) is None
    assert "operator examples use placeholders" in caplog.text




# ── the graph's facts in the reference (`base.graph_facts`) ──────────────────


def _facts(state: dict, protected=()) -> list[str]:
    from refactory_dsh.optim import base
    from refactory_dsh.optim.memory import state_view

    view = state_view(state)
    return base.graph_facts(view.node_states, view.edges, view.knobs,
                            nn_edits.catalogue("broad"), _check(state, protected))


def test_the_facts_name_what_sees_task_gives_on_this_pipeline():
    """A branch that `sees: ["task"]` gets the question once (prepare's user_prompt), never
    compare's prompt fact or the item record: the facts list exactly that, in the words
    the operator line uses."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import TASK_MEANING

    state = _start_state()
    names = sorted(name for _kind, name in core_ops._task_inputs(
        Graph.from_state(state["graph"]), state))
    assert names == ["user_prompt"]
    assert ('- `"sees": ["task"]` gives a new step user_prompt, where it is there before the '
            f"step: {TASK_MEANING}.") in _facts(state)


def test_the_facts_say_what_a_new_branch_of_a_fork_can_see():
    assert ("- fork `step_fork`: a new branch here can see: compare_prompt@prepare (+knob "
            "compare_task), hypotheses_prompt@notes_prompts (+knob hypotheses_task), "
            "precision_prompt@notes_prompts (+knob precision_task), user_prompt@prepare, "
            "compare_notes.txt@compare, current_item.json@data; \"task\" gives user_prompt."
            ) in _facts(_start_state())


def test_the_facts_give_each_refusal_in_the_attempts_own_words():
    facts = "\n".join(_facts(_start_state()))
    compare = next(line for line in facts.splitlines() if line.startswith("- `compare`:"))
    assert "remove is refused (" in compare and "leaves steps without an input only it makes" \
        in compare, compare
    # Every refusal said is the attempt's own: each listed edit is refused by it.
    state = _start_state()
    check = _check(state)
    assert isinstance(check({"op": "remove", "step": "compare"}), list)
