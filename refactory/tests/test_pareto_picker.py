"""The ``pareto`` parent picker: GEPA's per-item frontier as a core building block.

The frontier code lives once, in ``refactory_dsh/optim/pareto.py``; the default loop
reaches it as ``parent_picker: pareto`` (``optim.policies.ParetoSelector``) and the
GEPA strategy imports the same functions (its identical-trace test is in
``test_gepa.py``). Each test fails when the rule it names is reverted.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import gepa_ops
from refactory_dsh import nn, optim, profiles
from refactory_dsh import task as task_module
from refactory_dsh.loop import Loop, LoopConfig, opaque_item_key
from refactory_dsh.optim import Candidate, Optimizer, Proposal, ProposalContext, pareto
from refactory_dsh.optim.policies import HillClimbSelector, ParetoSelector

from test_gepa import _patch_mode

MODE_ID = "pareto_probe/tokens"


class TokenTask(gepa_ops.PlantedTask):
    """An item passes when the instruction holds its token, ``<id>``: each candidate's
    per-item scores are exactly the items it names."""

    name = "pareto_token_task"

    def verify(self, instance, workspace, **kw):  # type: ignore[no-untyped-def]
        from refactory_dsh.task import VerifyResult

        path = Path(workspace) / "knobs.json"
        text = (json.loads(path.read_text()) if path.exists() else {}).get("instruction") or ""
        passed = f"<{instance.id}>" in text
        return VerifyResult(passed=passed, score=1.0 if passed else 0.0,
                            details="" if passed else f"needs <{instance.id}>")


task_module.register_task(TokenTask.name, TokenTask)


class Tokens(nn.Mode):
    name = "tokens"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"instruction": nn.Knob(default="", kind="str", description="instruction")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                          task_ref=TokenTask.name, produces=("item_scores",)))
        graph.add(nn.Node("work", "op", call="gepa_ops:write_knobs", writes=("knobs.json",),
                          produces=("knobs_seen",), knobs=("instruction",)))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best", writes=("best.json",),
                          produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [e["score"] for e in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


#: The generalist names four of the six val items (mean 4/6); the specialist names
#: the other two (mean 2/6), which the generalist misses; the copy names v1 alone and
#: is dominated by the generalist.
GENERALIST = "<v1><v2><v3><v4>"
SPECIALIST = "<v5><v6>"
DOMINATED = "<v1>"


class Scripted(Optimizer):
    """Writes the three texts, then texts that name nothing, each from the parent the
    picker chose (``set_knob`` replaces the whole text, so the parent does not change
    what the child scores)."""

    name = "scripted"

    def __init__(self) -> None:
        super().__init__(k=1)
        self.parents: list[str] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        step = len(self.parents)
        parent = parents[0]
        self.parents.append(parent.state["knobs"]["instruction"]["value"])
        text = [GENERALIST, SPECIALIST, DOMINATED][step] if step < 3 else f"nothing {step}"
        return [Proposal(edit={"op": "set_knob", "name": "instruction", "value": text},
                         parent=parent.id)]


def _run(tmp_path: Path, picker: str, *, epochs: int = 14, seed: int = 3,
         selection: str = "") -> tuple[Loop, Scripted]:
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"train": gepa_ops.TRAIN_IDS, "val": gepa_ops.VAL_IDS}))
    proposer = Scripted()
    cfg = LoopConfig(epochs=epochs, batch_size=1, optimizer="random", parent_picker=picker,
                     seed=seed, runs_root=tmp_path / "run", allowed_ops=("set_knob",),
                     parts_manifest=str(parts), max_evaluations=100, holdout=0.0,
                     val_gate=True, selection=selection)
    with _patch_mode(Tokens):
        loop = Loop(MODE_ID, cfg, project=Path("."), optimizer=proposer)
        loop.run()
    return loop, proposer


def _by_text(loop: Loop) -> dict[str, Candidate]:
    return {c.state["knobs"]["instruction"]["value"]: c for c in loop.population}


# ── the default loop picks a specialist with a lower mean ──────────────────


def test_default_loop_with_pareto_breeds_from_a_lower_mean_specialist(tmp_path):
    loop, proposer = _run(tmp_path, "pareto")
    assert loop.selection_mode() == "pool", "a population picker defaults to pool"
    texts = _by_text(loop)
    gate = loop._gate()
    assert gate.cache[texts[SPECIALIST].id] < gate.cache[texts[GENERALIST].id]
    assert texts[SPECIALIST].id in loop._gate_rejected, "setup: the gate rejected it"
    later = proposer.parents[3:]
    # The specialist is best on v5 and v6, so it is on the frontier and is bred from,
    # though its val mean is half the generalist's and the gate rejected it.
    assert SPECIALIST in later
    assert GENERALIST in later
    # The dominated copy, the seed and the empty children are never parents.
    assert set(later) <= {GENERALIST, SPECIALIST}


def test_hill_climb_never_breeds_from_the_specialist(tmp_path):
    """The contrast: a mean-ranked picker on the same run (the same pool) never picks it."""
    loop, proposer = _run(tmp_path, "hill_climb", selection="pool")
    assert SPECIALIST not in proposer.parents[3:]
    assert set(proposer.parents[3:]) == {GENERALIST}


def test_the_draw_is_weighted_by_items_won_and_uses_the_loops_item_keys(tmp_path):
    loop, _proposer = _run(tmp_path, "pareto", epochs=4)
    texts = _by_text(loop)
    parents = loop._gated_parents()
    ctx = ProposalContext(val_scores=loop._val_scores(), val_item_scores=loop._val_item_scores())
    secret = loop.item_key_secret
    assert ctx.val_item_scores[texts[SPECIALIST].id] == {
        opaque_item_key(i, secret): (1.0 if i in ("v5", "v6") else 0.0)
        for i in gepa_ops.VAL_IDS}
    picker = ParetoSelector(seed=0)
    draws = Counter(
        picker.select(replace(ctx, generation=g), 1, parents)[0].id for g in range(2000))
    assert set(draws) == {texts[GENERALIST].id, texts[SPECIALIST].id}
    # Four items to two.
    assert 0.28 < draws[texts[SPECIALIST].id] / 2000 < 0.39
    assert picker.wins == {texts[GENERALIST].id: 4, texts[SPECIALIST].id: 2}
    # The reason the request gets names the frontier, not how many val items it wins.
    assert picker.reason(texts[SPECIALIST]) == (
        "on the per-item Pareto frontier of validation: one of 2 candidate(s) on it")


# ── the picker on its own ──────────────────────────────────────────────────


def _cands(n: int) -> list[Candidate]:
    out = []
    for index in range(n):
        c = Candidate.from_state({"knobs": {"x": {"value": index}}})
        c.score, c.evaluated = 0.0, True
        out.append(c)
    return out


def test_frontier_prunes_dominated_and_keeps_ties():
    a, b, c, d = _cands(4)
    item_scores = {
        a.id: {"k1": 1.0, "k2": 1.0, "k3": 0.0},
        b.id: {"k1": 1.0, "k2": 0.0, "k3": 0.0},   # dominated by a
        c.id: {"k1": 0.0, "k2": 0.0, "k3": 1.0},   # best on k3 alone
        d.id: {"k1": 0.0, "k2": 1.0, "k3": 1.0},   # ties a on k2, c on k3
    }
    ctx = ProposalContext(val_item_scores=item_scores, val_scores={
        cid: pareto.aggregate(s) for cid, s in item_scores.items()})
    picker = ParetoSelector(seed=1)
    chosen = {p.id for g in range(300)
              for p in picker.select(replace(ctx, generation=g), 2, [a, b, c, d])}
    assert b.id not in chosen
    # c is dominated by d (every front holding c holds d), a and d stay.
    assert chosen == {a.id, d.id}
    fronts = pareto.build_fronts([item_scores[x.id] for x in (a, b, c, d)])
    assert fronts == {"k1": {0, 1}, "k2": {0, 3}, "k3": {2, 3}}
    means = [pareto.aggregate(item_scores[x.id]) for x in (a, b, c, d)]
    assert pareto.remove_dominated_programs(fronts, means) == {"k1": {0}, "k2": {0, 3}, "k3": {3}}


def test_draws_are_a_function_of_seed_and_step():
    a, b = _cands(2)
    ctx = ProposalContext(epoch=2, generation=5, val_item_scores={
        a.id: {"k1": 1.0, "k2": 0.0}, b.id: {"k1": 0.0, "k2": 1.0}})
    first = [p.id for p in ParetoSelector(seed=4).select(ctx, 20, [a, b])]
    assert first == [p.id for p in ParetoSelector(seed=4).select(ctx, 20, [a, b])]
    # The picker's own generator (ParentSelector.rng), not the global one.
    random.seed(0)
    again = [p.id for p in ParetoSelector(seed=4).select(ctx, 20, [a, b])]
    assert again == first
    assert first != [p.id for p in ParetoSelector(seed=5).select(ctx, 20, [a, b])]


def test_without_item_scores_it_breeds_from_the_best_by_val():
    a, b = _cands(2)
    ctx = ProposalContext(val_scores={a.id: 0.2, b.id: 0.7})
    picker = ParetoSelector(seed=0)
    assert [p.id for p in picker.select(ctx, 3, [a, b])] == [b.id] * 3
    assert "no per-item validation scores" in picker.reason(b)


# ── settings ───────────────────────────────────────────────────────────────


def test_pareto_is_a_registered_population_picker():
    assert optim.resolve("pareto") is ParetoSelector
    assert "pareto" in optim.picker_names()
    assert optim.is_population("pareto")
    assert profiles.normalize({"parent_picker": "pareto"})["parent_picker"] == "pareto"
    assert "pareto" in profiles.SETTING_HELP["parent_picker"]
    assert not issubclass(HillClimbSelector, ParetoSelector)


def test_gepa_uses_the_core_frontier_functions():
    """One copy: GEPA's pool and draws are core's functions, not a private port."""
    import refactory_packages.gepa.loop as gepa_loop
    import refactory_packages.gepa.pareto as gepa_pareto

    assert gepa_loop.select_from_pareto_front is pareto.select_from_pareto_front
    assert gepa_loop.find_dominator_programs is pareto.find_dominator_programs
    for name in ("is_dominated", "remove_dominated_programs", "select_from_pareto_front",
                 "find_dominator_programs"):
        assert not hasattr(gepa_pareto, name) or getattr(gepa_pareto, name) is getattr(pareto, name)
    pool = gepa_pareto.ProgramPool()
    pool.add("c0", [None], {"a": 1.0, "b": 0.0}, 0)
    pool.add("c1", [0], {"a": 1.0, "b": 1.0}, 0)
    assert pool.front_programs == pareto.build_fronts(pool.subscores)
