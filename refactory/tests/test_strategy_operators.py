"""Every outer loop can take every core operator, or says it is text-only.

A strategy's proposals become candidates only through the shared attempt path
(`Run.attempt` → `Optimizer.attempt` → `nn.edits.apply`), so the operators, their
validation and their wiring reach every strategy there, built-in or from a package. These
tests build each registered strategy and hand it one instance of each operator. A strategy
that never changes structure (SkillOpt edits one skill's text) must say so itself
(`OuterLoop.text_only`) rather than be skipped silently, and a strategy with no model draws
the operators with their default texts (`edits.random_edit(operators=True)`).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh import strategies
from refactory_dsh.loop import LoopConfig
from refactory_dsh.nn import Knob, core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, Proposal
from refactory_dsh.outerloop import Run

#: A registered mode whose graph takes every core operator (a fork, agent steps, a chain).
MODE = "livemath/skill_graph"


def _accepted(run: Run, seed: Candidate, op: str, rng: random.Random) -> Candidate | None:
    """A candidate made by one instance of ``op``, or None when no instance is accepted."""
    for _ in range(80):
        edit = core_ops.sample(op, seed.state, rng)
        if nn_edits.precheck(edit, seed.state) is None:
            outcome = run.attempt(Proposal(edit=edit, parent=seed.id), seed, record=False)
            if isinstance(outcome, Candidate):
                return outcome
    # An operator whose new steps no later prompt here can take alone (copies of a step
    # making no fact) is taken in a composition that gives them a reader (E-unread).
    from refactory_dsh.optim import base

    graph = seed.state["graph"]
    for _title, bundle in base.composition_examples(
            nn_edits.catalogue("broad"), list(graph["nodes"].values()), list(graph["edges"]),
            knob_values={k: v["value"] for k, v in seed.state["knobs"].items()}):
        if bundle[0]["op"] == op:
            outcome = run.attempt(Proposal(edits=tuple(bundle), parent=seed.id), seed,
                                  record=False)
            if isinstance(outcome, Candidate):
                return outcome
    return None


@pytest.mark.parametrize("name", strategies.available(Path(".")))
def test_every_strategy_takes_every_core_operator_or_declares_itself_text_only(name, tmp_path):
    cls = strategies.resolve(name, Path("."))
    if cls.text_only:
        # Declared on the strategy itself, with a reason, and its own edits are text only.
        assert "text_only" in vars(cls) and len(cls.text_only) > 20, name
        assert not set(cls.required_ops) & set(nn_edits.CORE_OPS), cls.required_ops
        return
    loop = strategies.create(name, MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random",
                                                    seed=1, runs_root=tmp_path,
                                                    edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    loop.population.append(seed)
    run = Run(_loop=loop)
    rng = random.Random(f"operators:{name}")
    taken = {op for op in nn_edits.CORE_OPS
             if (child := _accepted(run, seed, op, rng)) is not None and child.state != seed.state}
    assert taken == set(nn_edits.CORE_OPS), sorted(set(nn_edits.CORE_OPS) - taken)


def test_the_text_only_strategies_are_skillopt_and_gepa():
    """A strategy turning text-only is a decision, so the list of them is held here."""
    assert sorted(name for name in strategies.available(Path("."))
                  if strategies.resolve(name, Path(".")).text_only) == ["gepa", "skillopt"]


def test_a_strategy_with_no_model_draws_every_core_operator():
    """The random sampler, with operators on, reaches each core operator on the fuzz seeds
    within a seeded budget; without them it draws what it always did (none of them but the
    written-out loop and split)."""
    reached: set[str] = set()
    plain: set[str] = set()
    # fuzz_data_pass for fallback, which answers only for a step in a per-item pass.
    for name in ("fuzz_draft_review", "fuzz_fork_join", "fuzz_dataflow", "fuzz_data_pass"):
        state = sf.cached_applier(name).seed_state
        rng, rng_plain = random.Random(f"ops:{name}"), random.Random(f"plain:{name}")
        for _ in range(300):
            edit = nn_edits.random_edit(state, preset="broad", rng=rng, operators=True)
            if edit is not None:
                reached.add(edit["op"])
            edit = nn_edits.random_edit(state, preset="broad", rng=rng_plain)
            if edit is not None:
                plain.add(edit["op"])
    assert set(nn_edits.CORE_OPS) <= reached, sorted(set(nn_edits.CORE_OPS) - reached)
    assert not plain & (set(nn_edits.CORE_OPS) - {"loop", "split"}), plain


def test_a_sampled_operator_leaves_its_texts_to_the_defaults():
    """No placeholder from an operator's example (``<what the new step does>``) reaches a
    sampled edit: each text is left out, so the operator's default text is used."""
    state = sf.cached_applier("fuzz_fork_join").seed_state
    rng = random.Random("texts")
    for op in nn_edits.CORE_OPS:
        edit = core_ops.sample(op, state, rng)
        assert edit["op"] == op and "<" not in json.dumps(edit), edit


def test_a_graph_with_no_model_step_draws_no_operator():
    """The steps an operator adds by default are model steps, which a mode that never calls a
    model cannot run, so there the draws stay as they were."""
    from refactory_dsh import registry

    _package, mode = registry.resolve(Path("."), "circle_packing/adaptive")
    state = mode().state_dict()
    rng = random.Random("no model")
    drawn = {(nn_edits.random_edit(state, preset="broad", rng=rng, operators=True) or {}).get("op")
             for _ in range(200)}
    assert not drawn & (set(nn_edits.CORE_OPS) - {"loop", "split"}), drawn


def _best_of_n(mode: str, tmp_path: Path, preset: str = "broad"):
    loop = strategies.create("best_of_n", mode, LoopConfig(epochs=1, batch_size=1,
                                                           optimizer="random", seed=1,
                                                           runs_root=tmp_path, edit_preset=preset),
                             project=Path("."))
    return loop, Candidate.from_state(loop.mode.state_dict())


def test_best_of_n_without_a_model_makes_every_other_variant_structural(tmp_path):
    loop, seed = _best_of_n(MODE, tmp_path)
    variants = [loop._variant(index, seed) for index in range(4)]
    assert [v["op"] in nn_edits.CORE_OPS for v in variants] == [False, True, False, True], variants
    assert variants[0]["op"] == variants[2]["op"] == "set_knob"


def test_best_of_n_knob_variants_are_what_they_were_where_no_operator_applies(tmp_path):
    """Under a preset with no operator, or in a mode with no model step, every variant is the
    knob sample it always was: the structural draw has a random stream of its own."""
    for mode, preset in ((MODE, "knob_only"), ("circle_packing/adaptive", "broad")):
        loop, seed = _best_of_n(mode, tmp_path / preset, preset)
        knobs = [k for k, e in seed.state["knobs"].items() if not e["spec"].get("frozen")]
        for index in range(4):
            name = knobs[index % len(knobs)]
            expected = Knob.from_spec(seed.state["knobs"][name]["spec"]).sample(
                random.Random(f"best_of_n:1:{index}"), seed.state["knobs"][name]["value"])
            assert loop._variant(index, seed) == {"op": "set_knob", "name": name,
                                                  "value": expected}, (mode, index)


def test_rebase_replays_one_candidates_edits_onto_another_or_refuses_whole():
    """The structural merge's seam: B's own edits onto A's state; an edit that no longer
    applies there refuses the whole rebase, naming it, and A is left as it was."""
    state = sf.cached_applier("fuzz_draft_review").seed_state
    onto = nn_edits.apply({"op": "insert", "after": "draft", "kind": "agent"}, state)
    edits = [{"op": "loop", "step": "review", "judge": "agent"},
             {"op": "rewire", "step": "finalize", "sees": ["task", "review"]}]
    rebased = nn_edits.rebase(edits, onto)
    assert set(onto["graph"]["nodes"]) < set(rebased["graph"]["nodes"])
    before = dict(onto)
    with pytest.raises(nn_edits.EditError, match=r"edit 2 of 2 \(remove\)"):
        nn_edits.rebase([edits[0], {"op": "remove", "step": "no_such_step"}], onto)
    assert onto == before


def test_best_of_n_knob_variants_still_go_through_every_movable_knob(tmp_path):
    """With every other variant structural, the knob variants are counted on their own: twice
    as many variants as movable knobs give each knob one."""
    loop, seed = _best_of_n(MODE, tmp_path)
    movable = [k for k, e in seed.state["knobs"].items() if not e["spec"].get("frozen")]
    variants = [loop._variant(index, seed) for index in range(2 * len(movable))]
    varied = [v["name"] for v in variants if v["op"] == "set_knob"]
    assert sorted(varied) == sorted(movable), (varied, movable)
