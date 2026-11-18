"""``insert``'s 'takes_over' renames only what the renaming reaches.

Taking a step over renames its outputs for the steps before the new one. That reaches a
model step (its files are handed to it, its facts filled, under the names it declares),
an op or gate whose callable works from its declaration (`nn.by_declaration`: LiveMath
S4's notes_prompts and decide_prompt, the block ops), and code: the run records what it
returns under the new names (`graph.Node.renames`), its code as written, so a later
rewrite of the code cannot undo the rename. It never reaches a callable that opens or
returns a name of its own (S4's prepare and extract_answer), nor one whose behaviour
depends on the names it reads (`nn.names_matter`: select): those are refused.

- **The fuzz walk's finding** on S4: ``parallelize`` after step_join, then a step taking
  compare over -- once accepted into a graph that failed every item -- now builds a graph
  that runs, wherever the new step goes; on the walker, and on the spine for a step
  right after step_join.
- **Renames survive a rewrite**: a code step taken over, and a copy of a code step, run
  on both runtimes after their code knob is set to plain code returning the old names.
- **Refusals**: an op that opens or returns names of its own, and select.
- **Prompts unchanged**: the LiveMath ops now reading their declaration send the same
  prompts on the walker and the spine (`test_livemath_skill_graph` pins the walker's).
- **Any graph**: seeded walks over every fuzz seed mixing 'takes_over' with parallelize,
  serialize, branch, loop (and 'back_to'), route, fallback, split and merge, in both
  orders; every state accepted keeps the harness's invariants on the walker.
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

import structure_fuzz as sf
import structure_fuzz_graphs as fg
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate
from test_livemath_blocks import _spine, needs_spine

S4 = "livemath_skill_graph"
#: The fuzz walk's first edit: two new steps side by side after S4's fork.
PARALLEL = {"op": "parallelize", "after": "step_join",
            "new": [{"kind": "agent", "text": "Try A."}, {"kind": "code"}]}
#: The invariants a state the harness accepts keeps (the run ones on the walker).
CHECKED = ("lowering", "walker", "required", "dataflow", "forks", "knobs")
#: Code that returns the branch's output under the name it was built with.
PLAIN_CODE = "def main(inp):\n    return {'facts': {'%s': 'rewritten'}}\n"


def _applier(name: str = S4) -> sf.Applier:
    return sf.Applier(sf.seeds()[name])


def _accepted(applier: sf.Applier, edits: list[dict]) -> dict:
    states, refusal = applier.replay(edits)
    assert refusal is None, refusal
    return states[-1].state


def _refusal(applier: sf.Applier, edits: list[dict]) -> str:
    _states, refusal = applier.replay(edits)
    assert refusal, edits
    return " ".join(refusal)


def _on_both(applier: sf.Applier, state: dict) -> None:
    assert sf.spine_eligible(state)
    assert sf.check_state(applier, state, walker=True, spine=True,
                          only=(*CHECKED, "spine")) == []


# ── the finding ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("after", ["step_join", "step_join_join", "step_join_branch1",
                                   "decide_prompt", "decide"])
@pytest.mark.parametrize("kind", ["agent", "code"])
def test_s4_taking_compare_over_after_a_new_fork_runs_on_the_walker(after, kind):
    """notes_prompts and decide_prompt read what they declare: renamed for them, they
    read compare's own notes, and decide sees the new step's (every item graded)."""
    applier = _applier()
    state = _accepted(applier, [PARALLEL, {"op": "insert", "after": after, "kind": kind,
                                           "takes_over": "compare"}])
    nodes = state["graph"]["nodes"]
    assert nodes["compare"]["writes"] == ["compare.txt"]
    assert nodes["notes_prompts"]["reads"] == ["compare.txt"]
    assert sf.check_state(applier, state, walker=True, only=CHECKED) == []


@needs_spine
def test_s4_a_step_after_step_join_taking_compare_over_runs_on_both_runtimes():
    applier = _applier()
    _on_both(applier, _accepted(applier, [
        {"op": "insert", "after": "step_join", "kind": "agent", "takes_over": "compare",
         "text": "Merge the notes above into one set of notes for the final step."}]))


def test_s4_taking_decide_prompt_over_runs_on_the_walker():
    """decide_prompt sets its declared fact, so it can be taken over (renamed)."""
    applier = _applier()
    state = _accepted(applier, [{"op": "insert", "after": "decide", "kind": "code",
                                 "takes_over": "decide_prompt"}])
    assert state["graph"]["nodes"]["decide_prompt"]["produces"] != ["decide_prompt"]
    assert sf.check_state(applier, state, walker=True, only=CHECKED) == []


# ── refusals ──────────────────────────────────────────────────────────────


def test_an_op_that_returns_its_outputs_by_name_cannot_be_taken_over():
    """S4's prepare returns {'user_prompt': ..., 'compare_prompt': ...} by name."""
    reason = _refusal(_applier(), [{"op": "insert", "after": "compare", "kind": "code",
                                    "takes_over": "prepare"}])
    assert "'prepare' runs" in reason and "names of its own" in reason, reason


def test_an_op_that_opens_a_renamed_file_by_name_refuses_it():
    """S4's extract_answer opens answer.txt by name: decide taken over after it is
    refused."""
    reason = nn_edits.precheck({"op": "insert", "after": "extract_answer", "kind": "code",
                                "takes_over": "decide"}, _applier().seed_state)
    assert reason and "'extract_answer' reads 'decide''s output 'answer.txt'" in reason, reason


def test_select_whose_names_matter_refuses_a_renamed_read():
    """select matches a judge's label against the names of the files it reads: renamed,
    a label naming compare_copy1 would match nothing."""
    applier = _applier()
    state = _accepted(applier, [
        {"op": "parallelize", "steps": ["compare"], "copies": 1},
        {"op": "insert", "after": "compare_join", "kind": "agent", "id": "choose",
         "text": "Name the better notes: compare or compare_copy1."},
        {"op": "insert", "after": "choose", "kind": "op", "id": "pick",
         "call": core_ops.select_call()}])
    assert "compare_copy1.txt" in state["graph"]["nodes"]["pick"]["reads"]
    reason = nn_edits.precheck({"op": "insert", "after": "pick", "kind": "code",
                                "takes_over": "compare_copy1"}, state)
    assert reason and "'pick' reads 'compare_copy1''s output 'compare_copy1.txt'" in reason, \
        reason


def test_a_renamed_read_an_op_would_match_by_its_built_name_is_refused():
    """With precision taken out, decide_prompt reads two of its three notes and finds
    them by their built names (`nn.declared_files`); hypotheses' notes renamed would be
    found by none, so the check -- resolving the reads as the run does -- refuses it. A
    take over decide_prompt does not read under a renamed name is still taken."""
    state = nn_edits.apply({"op": "remove", "step": "precision"}, _applier().seed_state)
    assert state["graph"]["nodes"]["decide_prompt"]["reads"] == [
        "compare_notes.txt", "hypotheses_notes.txt"]
    for target in ("hypotheses", "compare"):
        reason = nn_edits.precheck({"op": "insert", "after": "decide", "kind": "code",
                                    "takes_over": target}, state)
        assert reason and "'decide_prompt' reads" in reason, (target, reason)
    assert nn_edits.precheck({"op": "insert", "after": "hypotheses", "kind": "code",
                              "takes_over": "hypotheses"}, state) is None


def test_renames_is_the_edit_layer_s_own_field():
    """The optimizer is never shown ``renames`` and a proposal that writes it is refused."""
    from refactory_dsh.nn.node_schema import render_schema
    from refactory_dsh.optim import Proposal
    from refactory_dsh.optim.base import proposal_refusal

    assert "renames" not in "\n".join(render_schema())
    node = {"id": "c", "kind": "op", "call": "refactory_dsh.snippet:run", "produces": ["c"],
            "code": "def main(inp):\n    return {'facts': {'x': 1}}\n",
            "renames": {"facts": {"x": "c"}}}
    refused = proposal_refusal(Proposal(edit={"op": "insert_after", "target": "compare",
                                              "node": node}))
    assert refused and refused[0].startswith("E-node") and "'renames'" in refused[0], refused
    node.pop("renames")
    assert proposal_refusal(Proposal(edit={"op": "insert_after", "target": "compare",
                                           "node": node})) == []


def test_an_op_working_from_its_declaration_is_renamed_not_refused():
    """The same shape with ops marked `nn.by_declaration` (the fuzz seeds' step) is taken."""
    state = _applier("fuzz_draft_review").seed_state
    graph = sf._graph_of(state)
    readers = [n for n, node in graph.nodes.items() if node.kind == "op" and node.reads]
    assert readers and all(graph.nodes[n].call == fg.STEP for n in readers)
    for target, node in sorted(graph.nodes.items()):
        if node.kind != "op" or not node.writes:
            continue
        for after in sorted(sf._downstream_ids(graph, target) - {target}):
            edit = {"op": "insert", "after": after, "kind": "code", "takes_over": target}
            reason = nn_edits.precheck(edit, state)
            assert reason is None or "cannot rename" not in reason, (edit, reason)


# ── renames survive a rewrite of the code ─────────────────────────────────


#: A code branch of a new fork taken over after its join (on S4); and a code step that
#: names its outputs, split into a copy of itself and a review (the fuzz corpus case).
TAKEN = [PARALLEL, {"op": "insert", "after": "step_join_join", "kind": "code",
                    "takes_over": "step_join_branch2"}]
COPIED = json.loads((sf.CORPUS / "split_copies_code_that_names_its_outputs.json")
                    .read_text())["edits"]
#: Code returning both the fact and the file the copied step was built with.
COPY_CODE = "def main(inp):\n    return {'facts': {'code1_fact': 1}, 'files': {'code1.txt': 'x'}}\n"
REWRITES = [(S4, TAKEN, "step_join_branch2", PLAIN_CODE % "step_join_branch2"),
            ("fuzz_tool_step", COPIED, "code1_a", COPY_CODE)]


def test_a_code_step_taken_over_keeps_its_code_and_declares_its_renames():
    seed = _applier().replay([PARALLEL])[0][-1].state
    state = _accepted(_applier(), TAKEN)
    branch = state["graph"]["nodes"]["step_join_branch2"]
    renamed = "step_join_branch2_own_step_join_branch2"
    assert branch["produces"] == [renamed]
    assert branch["renames"] == {"facts": {"step_join_branch2": renamed}}
    assert state["graph"]["nodes"]["step_join_join_next"]["produces"] == ["step_join_branch2"]
    assert state["knobs"]["step_join_branch2__code"] == seed["knobs"]["step_join_branch2__code"]


@pytest.mark.parametrize("seed, edits, step, code", REWRITES, ids=["taken over", "copied"])
def test_code_rewritten_to_return_the_old_names_still_runs_on_the_walker(seed, edits, step, code):
    applier = _applier(seed)
    state = _accepted(applier, [*edits, {"op": "set_knob", "name": f"{step}__code",
                                         "value": code}])
    assert state["graph"]["nodes"][step]["renames"]
    assert sf.check_state(applier, state, walker=True, only=CHECKED) == []


@needs_spine
@pytest.mark.parametrize("seed, edits, step, code", REWRITES, ids=["taken over", "copied"])
def test_code_rewritten_to_return_the_old_names_still_runs_on_the_spine(seed, edits, step, code):
    applier = _applier(seed)
    _on_both(applier, _accepted(applier, [*edits, {"op": "set_knob", "name": f"{step}__code",
                                                   "value": code}]))


# ── the LiveMath ops send the same prompts ────────────────────────────────


def _exchanges(mode) -> tuple[dict, dict]:
    """Every agent exchange per item of ``mode`` with default knobs, on each runtime."""
    state = mode.state_dict()
    graph = sf._graph_of(state)
    answers = {n: f"{n} says <answer>B</answer>" for n, x in graph.nodes.items()
               if x.kind == "agent"}
    seed = sf.seeds()[S4]
    tmp = Path(tempfile.mkdtemp(prefix="take-over-prompts-"))
    walked = sf.run_walker(seed, state, tmp / "walker", sf.ScriptedModel(answers))
    assert walked.exit_ok, walked.error
    with sf.livemath_fixture():
        assert _spine(graph, tmp / "spine", tmp, agent_responses=answers, mode_state=state) == 0
    spun = json.loads((tmp / "spine" / "item_records.json").read_text())

    def calls(records):
        return {str(r["id"]): [(e["node"], e["prompt"]) for e in r.get("exchanges", [])]
                for r in records}
    return calls(walked.facts["item_scores"]), calls(spun)


@needs_spine
@pytest.mark.parametrize("block", ["skill_graph", "solve_then_verify",
                                   "two_attempts_then_judge", "solve_then_self_check"])
def test_the_ops_reading_their_declaration_send_the_same_prompts_on_both_runtimes(block):
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.skill_graph import SkillGraph

    walker, spine = _exchanges(SkillGraph() if block == "skill_graph" else Solver(block=block))
    assert walker == spine and sum(map(len, walker.values())) >= 10


# ── any graph ─────────────────────────────────────────────────────────────


SHAPES = ("parallelize", "serialize", "branch", "loop", "back_to", "route", "fallback",
          "split", "merge")
WALKS = 240


def _take_over(generator: sf.EditGenerator, state: dict, rng: random.Random) -> dict | None:
    graph = sf._graph_of(state)
    outputs = [n for n, node in graph.nodes.items()
               if node.kind in ("agent", "op") and (node.writes or node.produces)]
    anchors = [n for n, node in graph.nodes.items() if node.kind in ("agent", "op", "join")]
    if not outputs or not anchors:
        return None
    after = rng.choice(anchors)
    before = [n for n in outputs if n != after and after in sf._downstream_ids(graph, n)]
    spec = generator.step_spec(state, kinds=("agent", "code", "op"), sees=False)
    if spec.get("kind") not in ("agent", "code", "op"):
        spec = {"kind": "code"}
    return {"op": "insert", "after": after, **spec, "takes_over": rng.choice(before or outputs)}


def _shape(generator: sf.EditGenerator, state: dict, rng: random.Random) -> dict | None:
    graph = sf._graph_of(state)
    op = rng.choice(SHAPES)
    if op == "back_to":
        agents = sorted(n for n, node in graph.nodes.items() if node.kind == "agent")
        steps = sorted(n for n, node in graph.nodes.items() if node.kind in ("agent", "op"))
        if not agents:
            return None
        return {"op": "loop", "step": rng.choice(steps), "back_to": rng.choice(agents),
                "judge": fg.PROCEED}
    return generator.core_op(state, op, rng.choice(sorted(graph.nodes)))


def test_taking_over_mixed_with_every_shape_keeps_every_invariant():
    names = sorted(sf.seeds())
    taken, bad = 0, []
    for walk in range(WALKS):
        rng = random.Random(f"takes_over:{walk}")
        generator = sf.EditGenerator(rng, code_nodes=False)
        applier = _applier(names[walk % len(names)])
        current = Candidate.from_state(applier.seed_state)
        edits: list[dict] = []
        for step in rng.choice((("s", "t"), ("s", "s", "t"), ("t", "s"), ("s", "t", "s"))):
            edit = (_shape if step == "s" else _take_over)(generator, current.state, rng)
            if edit is None:
                continue
            outcome = applier.attempt(current, edit, {current.id})
            if isinstance(outcome, list):
                continue
            edits.append(edit)
            current = outcome
        if not any(e.get("takes_over") for e in edits):
            continue
        taken += 1
        found = sf.check_state(applier, current.state, walker=True, only=CHECKED)
        if found:
            bad.append((applier.seed.name, edits, [str(v) for v in found]))
    assert taken > WALKS // 3, taken
    assert not bad, bad[:3]


def _fallbacks_and_loops(state: dict) -> list[dict]:
    graph = sf._graph_of(state)
    steps = sorted(n for n, node in graph.nodes.items() if node.kind in ("agent", "op"))
    agents = sorted(n for n, node in graph.nodes.items() if node.kind == "agent")
    return [*({"op": "fallback", "step": s, "kind": "code"} for s in steps),
            *({"op": "loop", "step": s, "back_to": b, "judge": fg.PROCEED}
              for s in steps for b in agents)]


#: How many of the states each direction reaches are run (a seeded sample).
RUN_PAIRS = 20


def _sound(applier: sf.Applier, accepted: list[tuple[list[dict], dict]]) -> None:
    rng = random.Random(applier.seed.name)
    for edits, state in rng.sample(accepted, min(RUN_PAIRS, len(accepted))):
        found = sf.check_state(applier, state, walker=True, only=CHECKED)
        assert found == [], (edits, [str(v) for v in found])


@pytest.mark.parametrize("name", [S4, "fuzz_data_pass"])
def test_taking_over_with_a_fallback_or_a_loop_back_keeps_every_invariant(name):
    """Every fallback and 'back_to' loop a seed takes, then every take over the result
    takes; and every take over, then every fallback and loop back (each direction's
    accepted states statically, a seeded sample of them on the walker too)."""
    applier = _applier(name)
    seed = Candidate.from_state(applier.seed_state)
    shape_first: list[tuple[list[dict], dict]] = []
    for shape in _fallbacks_and_loops(applier.seed_state):
        shaped = applier.attempt(seed, shape, {seed.id})
        if isinstance(shaped, list):
            continue
        graph = sf._graph_of(shaped.state)
        for target in sorted(graph.nodes):
            for after in sorted(sf._downstream_ids(graph, target) - {target}):
                edit = {"op": "insert", "after": after, "kind": "code", "takes_over": target}
                taken = applier.attempt(shaped, edit, {shaped.id})
                if not isinstance(taken, list):
                    shape_first.append(([shape, edit], taken.state))
    take_first: list[tuple[list[dict], dict]] = []
    nodes = sorted(sf._graph_of(applier.seed_state).nodes)
    for target in nodes:
        for after in nodes:
            edit = {"op": "insert", "after": after, "kind": "code", "takes_over": target}
            taken = applier.attempt(seed, edit, {seed.id})
            if isinstance(taken, list):
                continue
            for shape in _fallbacks_and_loops(taken.state):
                shaped = applier.attempt(taken, shape, {taken.id})
                if not isinstance(shaped, list):
                    take_first.append(([edit, shape], shaped.state))
    assert shape_first and take_first
    for edits, state in shape_first + take_first:
        found = sf.check_state(applier, state, walker=False, only=CHECKED)
        assert found == [], (edits, [str(v) for v in found])
    _sound(applier, shape_first)
    _sound(applier, take_first)
