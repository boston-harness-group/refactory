"""``insert``'s 'takes_over' renames only what the renaming reaches.

Taking a step over renames its outputs for the steps before the new one. That reaches a
model step (its files are handed to it, its facts filled, under the names it declares),
an op or gate whose callable works from its declaration (`nn.by_declaration`), and code
(its code is wrapped to return under the new names, `core_ops._renamed_code`). It never
reaches a callable that opens or returns a name of its own: LiveMath S4's notes_prompts
opens ``compare_notes.txt`` and decide_prompt returns ``decide_prompt`` by name. Such a
take over is refused, never accepted into a graph that fails every item.

- **The fuzz walk's finding** on S4: ``parallelize`` after step_join, then a step taking
  compare over: refused, wherever the new step goes after notes_prompts.
- **Both runtimes**: what is accepted on S4 next to it -- the step after compare taking
  it over, and a step after a new fork taking over one of its code branches -- grades
  every item on the walker and on the spine, with the same facts.
- **Any graph**: seeded walks over every fuzz seed mixing 'takes_over' with parallelize,
  serialize, branch, loop (and 'back_to'), route, fallback, split and merge, in both
  orders; every state accepted keeps the harness's invariants on the walker.
"""
from __future__ import annotations

import random
import re

import pytest

import structure_fuzz as sf
import structure_fuzz_graphs as fg
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate
from test_livemath_blocks import needs_spine

S4 = "livemath_skill_graph"
#: The fuzz walk's first edit: two new steps side by side after S4's fork.
PARALLEL = {"op": "parallelize", "after": "step_join",
            "new": [{"kind": "agent", "text": "Try A."}, {"kind": "code"}]}
#: The invariants a state the harness accepts keeps (the run ones on the walker).
CHECKED = ("lowering", "walker", "required", "dataflow", "forks", "knobs")


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


# ── the finding ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("after", ["step_join", "step_join_join", "step_join_branch1",
                                   "decide_prompt", "decide"])
@pytest.mark.parametrize("kind", ["agent", "code"])
def test_s4_taking_compare_over_after_a_new_fork_is_refused(after, kind):
    """notes_prompts runs before the new step and opens compare_notes.txt by name:
    renamed for it, its read would find nothing (every item failed on both runtimes)."""
    reason = _refusal(_applier(), [PARALLEL, {"op": "insert", "after": after, "kind": kind,
                                              "takes_over": "compare"}])
    assert re.search(r"'(notes_prompts|decide_prompt)' reads 'compare''s output "
                     r"'compare_notes.txt'", reason), reason
    assert "cannot rename" in reason


def test_s4_taking_over_an_op_that_returns_its_outputs_by_name_is_refused():
    """decide_prompt's callable returns {'decide_prompt': ...}: it cannot be renamed."""
    reason = _refusal(_applier(), [{"op": "insert", "after": "decide", "kind": "code",
                                    "takes_over": "decide_prompt"}])
    assert "'decide_prompt' runs" in reason and "names of its own" in reason, reason


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


# ── both runtimes ─────────────────────────────────────────────────────────


#: Accepted next to the finding: compare taken over right after it (no step before the
#: new one reads it), and the new fork's code branch taken over after its join (its code
#: wrapped to return under the new names).
ACCEPTED = {
    "compare right after it": [PARALLEL, {"op": "insert", "after": "compare", "kind": "agent",
                                          "text": "Rewrite the notes.", "takes_over": "compare"}],
    "a code branch after the join": [PARALLEL, {"op": "insert", "after": "step_join_join",
                                                "kind": "code", "takes_over": "step_join_branch2"}],
}


@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_what_s4_accepts_runs_on_the_walker(name):
    applier = _applier()
    state = _accepted(applier, ACCEPTED[name])
    assert sf.check_state(applier, state, walker=True, only=CHECKED) == []


def test_a_code_step_taken_over_returns_its_outputs_under_the_new_names():
    state = _accepted(_applier(), ACCEPTED["a code branch after the join"])
    branch = state["graph"]["nodes"]["step_join_branch2"]
    renamed = "step_join_branch2_own_step_join_branch2"
    assert branch["produces"] == [renamed]
    assert state["graph"]["nodes"]["step_join_join_next"]["produces"] == ["step_join_branch2"]
    code = state["knobs"]["step_join_branch2__code"]["value"]
    assert f"{{'step_join_branch2': '{renamed}'}}" in code, code


@needs_spine
@pytest.mark.parametrize("name", sorted(ACCEPTED))
def test_what_s4_accepts_runs_on_the_spine_as_on_the_walker(name):
    applier = _applier()
    state = _accepted(applier, ACCEPTED[name])
    assert sf.spine_eligible(state)
    assert sf.check_state(applier, state, walker=True, spine=True,
                          only=(*CHECKED, "spine")) == []


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
