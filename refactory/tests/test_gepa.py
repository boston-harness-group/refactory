"""GEPA outer-loop strategy tests (``refactory_packages/gepa``).

The ported pieces (frontier, sampler, extractor, merge) are checked against GEPA
0.1.4's rules directly; the strategy end to end on a planted two-category task
(``tests/gepa_ops.py``) with a scripted reflection model.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

import gepa_ops
from refactory_dsh import nn
from refactory_dsh import strategies
from refactory_dsh import task as task_module
from refactory_dsh.loop import LoopConfig, opaque_item_key

from refactory_packages.gepa import merge as merge_mod
from refactory_packages.gepa.loop import GepaLoop, discover_components
from refactory_packages.gepa.pareto import (
    ProgramPool, find_dominator_programs, remove_dominated_programs, select_from_pareto_front,
)
from refactory_packages.gepa.reflection import (
    DEFAULT_PROMPT_TEMPLATE, INSTRUCTION_PLACEHOLDER, check_template, extract_instruction,
    feedback_text, reflection_prompt, reflective_dataset,
)
from refactory_packages.gepa.sampler import EpochShuffledBatchSampler
from refactory_packages.skillopt.trace import current_call_key

TASK_REF = gepa_ops.PlantedTask.name
MODE_ID = "gepa_probe/planted"
task_module.register_task(TASK_REF, gepa_ops.PlantedTask)


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("refactory_packages.skillopt.trace._sleep", lambda _s: None)


@pytest.fixture(autouse=True)
def _gepa_found_by_discovery():
    """Every test reaches ``gepa`` as the CLI does, by package discovery."""
    assert "gepa" not in strategies.STRATEGIES
    yield
    assert "gepa" not in strategies.STRATEGIES


# ── the planted modes ──────────────────────────────────────────────────────


def _graph(knobs: tuple[str, ...]) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                      task_ref=TASK_REF, produces=("item_scores",)))
    graph.add(nn.Node("work", "op", call="gepa_ops:write_knobs", writes=("knobs.json",),
                      produces=("knobs_seen",), knobs=knobs))
    graph.add(nn.Node("finish", "op", call="loop_ops:write_best", writes=("best.json",),
                      produces=("best",)))
    graph.connect("data", "finish")
    return graph


class OneKnob(nn.Mode):
    """One component, ``instruction``: a specialist in A or in B, never both."""

    name = "planted"
    seed_text = "Solve the item."

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"instruction": nn.Knob(default=cls.seed_text, kind="str", description="instruction")}

    def build(self) -> nn.Graph:
        return _graph(("instruction",))

    def score(self, run: nn.RunResult) -> float:
        scores = [e["score"] for e in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class TwoKnobs(OneKnob):
    """Two components, ``alpha_rule`` for A items and ``beta_rule`` for B items."""

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"alpha_rule": nn.Knob(default="Rule one.", kind="str", description="A rule"),
                "beta_rule": nn.Knob(default="Rule two.", kind="str", description="B rule")}

    def build(self) -> nn.Graph:
        return _graph(("alpha_rule", "beta_rule"))


class PerfectSeed(OneKnob):
    seed_text = "ALPHA"


def _patch_mode(mode_class: type[nn.Mode]):  # type: ignore[no-untyped-def]
    import refactory_dsh.registry as registry

    class FakePackage:
        name = "fake"
        description = ""

        def mode(self, name: str) -> type[nn.Mode]:
            return mode_class

    @contextlib.contextmanager
    def patched():  # type: ignore[no-untyped-def]
        original = registry.resolve
        registry.resolve = lambda project, identifier: (FakePackage(), mode_class)  # type: ignore[assignment]
        try:
            yield
        finally:
            registry.resolve = original  # type: ignore[assignment]

    return patched()


class FakeClient:
    """The reflection model: ``policy(prompt, call_key)`` answers each call."""

    def __init__(self, policy: Callable[[str, tuple | None], str]) -> None:
        self.policy = policy
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, system: str = "") -> str:
        key = current_call_key()
        self.calls.append({"prompt": prompt, "system": system, "call_key": key})
        return self.policy(prompt, key)


def follow_feedback(prompt: str, key: tuple | None) -> str:
    """Write what the feedback asks for most: a specialist in one category."""
    side = prompt.split("following are examples", 1)[1]
    asks = Counter(re.findall(r"must say (ALPHA|BETA)", side))
    word = asks.most_common(1)[0][0] if asks else "ALPHA"
    return f"Here it is:\n```text\nAlways answer {word}.\n```"


def _run(tmp_path: Path, mode: type[nn.Mode] = OneKnob, *, policy=follow_feedback,
         options: dict[str, Any] | None = None, max_rollouts: int = 80,
         parts: dict[str, list[str]] | None = None, **config: Any):  # type: ignore[no-untyped-def]
    manifest = parts or {"train": gepa_ops.TRAIN_IDS, "val": gepa_ops.VAL_IDS}
    parts_path = tmp_path / "parts.json"
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    parts_path.write_text(json.dumps(manifest))
    cfg = LoopConfig(**{
        "epochs": 100, "batch_size": 1, "optimizer": "random", "seed": 1,
        "runs_root": tmp_path / "run", "allowed_ops": ("set_knob",), "strategy": "gepa",
        "parts_manifest": str(parts_path), "max_evaluations": 100000,
        "max_rollouts": max_rollouts, **config,
    })
    client = FakeClient(policy)
    with _patch_mode(mode):
        loop = strategies.create("gepa", MODE_ID, cfg, strategy_options=dict(options or {}),
                                 project=Path("."), model=client)
        strategy = loop._custom_strategy
        assert isinstance(strategy, GepaLoop)
        report = loop.run()
    return strategy, report, client, loop


def _trace(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "run" / "gepa_trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decisions(tmp_path: Path, step: str) -> list[dict[str, Any]]:
    return [e for e in _trace(tmp_path) if e.get("type") == "decision" and e.get("step_type") == step]


def _texts(loop, strategy: GepaLoop, idx: int) -> dict[str, str]:  # type: ignore[no-untyped-def]
    cid = strategy.pool.candidate_ids[idx]
    cand = next(c for c in loop.population if c.id == cid)
    return {k: v["value"] for k, v in cand.state["knobs"].items()}


# ── the ported pieces ──────────────────────────────────────────────────────


def test_front_replaces_on_better_and_adds_on_tie():
    pool = ProgramPool()
    pool.add("c0", [None], {"a": 0.0, "b": 1.0}, 0)
    pool.add("c1", [0], {"a": 1.0, "b": 1.0}, 4)
    pool.add("c2", [0], {"a": 1.0, "b": 0.0}, 8)
    assert pool.front_programs == {"a": {1, 2}, "b": {0, 1}}
    assert pool.best() == 1
    # Program 0 and 2 are dominated by 1 (every front holding them holds 1).
    assert remove_dominated_programs(pool.front_programs, pool.aggregates()) == {"a": {1}, "b": {1}}
    assert find_dominator_programs(pool.front_programs, pool.aggregates()) == [1]
    reloaded = ProgramPool.load(json.loads(json.dumps(pool.state_dict())))
    assert reloaded.front_programs == pool.front_programs and reloaded.pointers == pool.pointers


def test_frontier_draw_is_weighted_by_instances_won():
    pool = ProgramPool()
    pool.add("c0", [None], {"a": 1.0, "b": 0.0, "c": 0.0}, 0)
    pool.add("c1", [0], {"a": 0.0, "b": 1.0, "c": 1.0}, 0)
    rng = random.Random(7)
    draws = Counter(select_from_pareto_front(pool.front_programs, pool.aggregates(), rng)
                    for _ in range(3000))
    assert set(draws) == {0, 1}
    assert 0.28 < draws[0] / 3000 < 0.39  # one instance in three


def test_sampler_pads_with_the_least_frequent_and_reshuffles_each_epoch():
    ids = ["a", "b", "c", "d", "e"]
    sampler = EpochShuffledBatchSampler(3, random.Random(0))
    expected = list(ids)
    random.Random(0).shuffle(expected)
    first = sampler.next_minibatch(ids, 0) + sampler.next_minibatch(ids, 1)
    assert first == expected + [expected[-1]]
    # Iteration 2 is the next epoch: a fresh shuffle of the same ids.
    assert set(sampler.next_minibatch(ids, 2)) <= set(ids)
    assert sampler.epoch == 1
    clone = EpochShuffledBatchSampler(3, random.Random(0))
    clone.load_state_dict(json.loads(json.dumps(sampler.state_dict())))
    assert clone.state_dict() == sampler.state_dict()


@pytest.mark.parametrize("reply,want", [
    ("```\nnew text\n```", "new text"),
    ("Sure:\n```markdown\nline one\nline two\n```\nDone.", "line one\nline two"),
    ("no fences at all", "no fences at all"),
    ("```python\nhalf open", "half open"),
    ("half closed\n```", "half closed"),
    ("```a``` and ```b```", "a``` and ```b"),
])
def test_extract_instruction_matches_gepa(reply, want):
    assert extract_instruction(reply) == want


def test_template_and_reflection_prompt():
    with pytest.raises(ValueError):
        check_template("no placeholders")
    records = [{"Inputs": "q", "Generated Outputs": "r", "Feedback": "f"}]
    prompt = reflection_prompt("OLD", records)
    assert prompt.startswith("I provided an assistant with the following instructions")
    assert "```\nOLD\n```" in prompt
    assert "# Example 1\n## Inputs\nq\n\n## Generated Outputs\nr\n\n## Feedback\nf\n\n" in prompt
    assert prompt.endswith("Provide the new instructions within ``` blocks.")
    assert DEFAULT_PROMPT_TEMPLATE.count("<curr_param>") == 1


def test_reflective_dataset_takes_one_call_of_the_components_nodes():
    records = [
        {"id": "t1", "score": 0.0, "fail_reason": "wrong",
         "exchanges": [{"node": "q", "prompt": "Do X.\nInput: 1", "response": "r1"},
                       {"node": "other", "prompt": "p", "response": "r"}]},
        {"id": "t2", "score": 1.0, "fail_reason": "", "exchanges": [{"node": "other"}]},
        {"id": "t3", "score": 1.0, "exchanges": [{"node": "q", "prompt": "Do X. 3", "response": "r3"}]},
    ]
    data = reflective_dataset(records, ["q"], "Do X.", feedback="details", rng=random.Random(0))
    assert data == [
        {"Inputs": f"{INSTRUCTION_PLACEHOLDER}\nInput: 1", "Generated Outputs": "r1", "Feedback": "wrong"},
        {"Inputs": f"{INSTRUCTION_PLACEHOLDER} 3", "Generated Outputs": "r3",
         "Feedback": "This trajectory got a score of 1.0."},
    ]
    assert feedback_text(records[0], "score") == "This trajectory got a score of 0.0."


def test_merge_takes_each_descendants_change():
    texts = [{"a": "x", "b": "y"}, {"a": "A", "b": "y"}, {"a": "x", "b": "B"}]
    parents = [[None], [0], [0]]
    performed: tuple[list, list] = ([], [])
    found = merge_mod.sample_merge([0.0, 0.5, 0.5], random.Random(0), [1, 2], performed, texts, parents)
    assert found is not None
    merged, id1, id2, ancestor = found
    assert (merged, id1, id2, ancestor) == ({"a": "A", "b": "B"}, 1, 2, 0)
    assert performed[1] == [(1, 2, (1, 2))]
    performed[0].append((1, 2, 0))
    assert merge_mod.sample_merge([0.0, 0.5, 0.5], random.Random(0), [1, 2], performed, texts,
                                  parents) is None


def test_eval_subsample_draws_from_each_side():
    s1 = {"k1": 1.0, "k2": 1.0, "k3": 0.0, "k4": 0.0, "k5": 1.0, "k6": 0.5}
    s2 = {"k1": 0.0, "k2": 0.0, "k3": 1.0, "k4": 1.0, "k5": 1.0, "k6": 0.5}
    picked = merge_mod.select_eval_subsample(s1, s2, random.Random(3))
    assert len(picked) == 5 and len(set(picked)) == 5
    assert len([k for k in picked if s1[k] > s2[k]]) == 2
    assert len([k for k in picked if s2[k] > s1[k]]) == 2


def test_components_are_the_per_node_instructions():
    state = {
        "knobs": {
            "skill": {"value": "", "spec": {"kind": "str"}},
            "q_instruction": {"value": "", "spec": {"kind": "str"}},
            "q__system_prompt": {"value": "", "spec": {"kind": "str"}},
            "s_instruction": {"value": "", "spec": {"kind": "str"}},
            "frozen": {"value": "", "spec": {"kind": "str", "frozen": True}},
            "k": {"value": 3, "spec": {"kind": "int"}},
        },
        "graph": {"nodes": {
            "q": {"kind": "agent", "knobs": ["skill", "q_instruction", "q__system_prompt"],
                  "field_bindings": [["system_prompt_template", "q__system_prompt"]]},
            "s": {"kind": "agent", "knobs": ["skill", "s_instruction", "frozen"]},
            "op": {"kind": "op", "knobs": ["k"]},
        }},
    }
    assert discover_components(state, None) == [("q_instruction", ["q"]), ("s_instruction", ["s"])]
    assert discover_components(state, ("skill",)) == [("skill", ["q", "s"])]
    with pytest.raises(ValueError):
        discover_components(state, ("frozen",))


# ── the strategy end to end ────────────────────────────────────────────────


def test_val_item_scores_are_opaque_per_item(tmp_path):
    strategy, report, _client, loop = _run(tmp_path, max_rollouts=40)
    seed_id = strategy.pool.candidate_ids[0]
    scores = strategy.pool.subscores[0]
    assert sorted(scores) == sorted(opaque_item_key(i) for i in gepa_ops.VAL_IDS)
    assert not set(scores) & set(gepa_ops.VAL_IDS)
    assert all(v == 0.0 for v in scores.values())  # the seed says neither word
    assert seed_id == loop.population[0].id


def test_search_finds_both_specialists_and_keeps_them_on_the_frontier(tmp_path):
    strategy, report, client, loop = _run(tmp_path, max_rollouts=120)
    texts = {_texts(loop, strategy, i)["instruction"] for i in range(len(strategy.pool.candidate_ids))}
    assert {"Always answer ALPHA.", "Always answer BETA."} <= texts
    # Each specialist wins half of val; neither dominates, so both stay on the frontier.
    assert len(find_dominator_programs(strategy.pool.front_programs, strategy.pool.aggregates())) == 2
    assert report["strategy_report"]["best_pareto_mean"] == pytest.approx(0.5)
    for d in _decisions(tmp_path, "reflect"):
        assert d["accepted"] == (d["child_sum"] > d["parent_sum"])
    # The reflection model saw train feedback only, never a val item.
    for call in client.calls:
        assert call["system"] == ""
        assert not any(re.search(rf"\b{v}\b", call["prompt"]) for v in gepa_ops.VAL_IDS)
    assert strategy.best_candidate(loop.population).id == strategy.pool.candidate_ids[strategy.pool.best()]


def test_equal_minibatch_sum_is_rejected(tmp_path):
    strategy, report, client, _ = _run(
        tmp_path, policy=lambda p, k: f"```\nSolve the item, attempt {k[1]}.\n```", max_rollouts=40)
    assert client.calls
    assert len(strategy.pool.candidate_ids) == 1
    decisions = _decisions(tmp_path, "reflect")
    assert decisions and all(not d["accepted"] and d["child_sum"] == d["parent_sum"] for d in decisions)


def test_round_robin_alternates_components(tmp_path):
    _strategy, _report, client, _ = _run(
        tmp_path, TwoKnobs, policy=lambda p, k: f"```\nnothing useful {k[1]}\n```", max_rollouts=60)
    order = [call["call_key"][2] for call in client.calls]
    assert len(order) >= 4
    assert order == [i % 2 for i in range(len(order))]


def test_module_selector_all_proposes_every_component(tmp_path):
    _strategy, _report, client, _ = _run(
        tmp_path, TwoKnobs, policy=lambda p, k: f"```\nnothing useful {k[1]}\n```",
        options={"module_selector": "all"}, max_rollouts=30)
    first = [c["call_key"] for c in client.calls if c["call_key"][1] == 0]
    assert [k[2] for k in first] == [0, 1]


def test_perfect_minibatches_are_skipped(tmp_path):
    parts = {"train": ["t1", "t3", "t5", "t7"], "val": ["v1", "v3"]}  # all category A
    strategy, report, client, _ = _run(tmp_path, PerfectSeed, parts=parts, max_rollouts=30)
    assert client.calls == []
    assert strategy.stats.get("skipped_perfect", 0) >= 3


def test_budget_stops_the_search(tmp_path):
    strategy, report, _client, loop = _run(tmp_path, max_rollouts=30)
    assert report["bound"] == "budget_exhausted"
    rollouts = sum(1 for _ in loop.population)  # candidates exist
    assert rollouts >= 1
    # The budget is checked at the top of an iteration: at most one iteration over.
    assert strategy.metric_calls <= 30 + 2 * 3 + len(gepa_ops.VAL_IDS)


def test_pareto_set_train_holds_out_train_ids_and_gates_on_val(tmp_path, monkeypatch):
    from refactory_dsh.outerloop import Run

    seen: list[tuple[str, tuple | None]] = []
    original = Run.evaluate

    def spy(self, candidate, *, part="train", items=None, seed=None, **kw):  # type: ignore[no-untyped-def]
        seen.append((part, tuple(items) if items else None))
        return original(self, candidate, part=part, items=items, seed=seed, **kw)

    monkeypatch.setattr(Run, "evaluate", spy)
    strategy, report, _client, loop = _run(tmp_path, options={"pareto_set": "train"},
                                           max_rollouts=80)
    held = set(strategy.pareto_ids)
    assert len(held) == 4 and held <= set(gepa_ops.TRAIN_IDS)
    feedback = [items for part, items in seen if part == "train" and items and len(items) <= 3]
    assert feedback and all(not set(items) & held for items in feedback)
    assert set(strategy.pool.subscores[0]) == held
    assert [p for p, _ in seen].count("val") >= 1
    checks = _decisions(tmp_path, "val_check")
    best = strategy.best_candidate(loop.population)
    if checks:
        assert best.id in (checks[-1]["candidate"], strategy.pool.candidate_ids[0])


def test_merge_runs_and_scores_on_the_pareto_set(tmp_path):
    def policy(prompt: str, key: tuple | None) -> str:
        return "```\nALPHA\n```" if key[2] == 0 else "```\nBETA\n```"

    strategy, report, _client, _ = _run(tmp_path, TwoKnobs, policy=policy,
                                        options={"use_merge": True}, max_rollouts=150)
    assert report["strategy_report"]["best_pareto_mean"] == pytest.approx(1.0)
    for d in _decisions(tmp_path, "merge"):
        assert d["accepted"] == (d["child_sum"] >= max(d["parent_sums"]))


def test_state_round_trips_and_resumes_the_same_draws(tmp_path):
    strategy, _report, _client, _ = _run(tmp_path, max_rollouts=50)
    state = json.loads(json.dumps(strategy.strategy_state_dict()))
    clone = GepaLoop()
    clone.configure(model=None)
    clone.load_strategy_state(state)
    assert clone.strategy_state_dict() == state
    assert clone.rng.random() == strategy.rng.random()
    assert clone.pool.front_programs == strategy.pool.front_programs


def test_options_are_checked():
    with pytest.raises(TypeError):
        GepaLoop().configure(model=None, candidate_selection_strategy="epsilon")
    with pytest.raises(TypeError):
        GepaLoop().configure(model=None, pareto_size=3)
    with pytest.raises(ValueError):
        GepaLoop().configure(model=None, reflection_prompt_template="no placeholders")
