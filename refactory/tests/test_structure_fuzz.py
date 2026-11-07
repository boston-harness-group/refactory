"""The structure-edit invariant harness (`structure_fuzz`): tiers, corpus and mutations.

- **Quick tier** (every run): a few hundred seeded random edits over every seed graph,
  through the outer loop's attempt path; every accepted state lowers, runs on the
  walker to a terminal, and keeps the static invariants.
- **Long tier** (``--run-structure-fuzz-long``): thousands of edits per seed, with
  a spine run beside the walker on a sample of the accepted states. Set
  ``STRUCTURE_FUZZ_EDITS`` and ``STRUCTURE_FUZZ_SEEDS`` to change its size, and
  ``STRUCTURE_FUZZ_REPORT`` to a path for its counts as JSON.
- **Regression corpus** (every run): each minimal case the harness ever found
  (``fixtures/structure_fuzz``) replays clean: refused, or accepted and sound.
- **Mutation checks**: each invariant's protection is switched off in turn, and the
  harness must find a violation of that invariant.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh.nn import edits as edits_mod
from refactory_dsh.optim import base as base_mod
from test_livemath_blocks import needs_spine

# `nn` exports the function `validate`, which shadows the module's name on the package.
validate_mod = importlib.import_module("refactory_dsh.nn.validate")

QUICK_SEED = 20261107
QUICK_EDITS = 300


def _report(stats: sf.Stats) -> str:
    lines = [json.dumps(stats.summary(), indent=1)]
    for finding in stats.findings[:10]:
        lines.append(f"{finding.seed}: {[str(v) for v in finding.violations][:3]}")
        lines.append(f"  edits: {json.dumps(finding.edits, default=str)[:1500]}")
    return "\n".join(lines)


# ── the seeds ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(sf.seeds()))
def test_every_seed_is_sound_before_any_edit(name, tmp_path):
    """The harness starts from graphs that keep every invariant themselves."""
    seed = sf.seeds()[name]
    applier = sf.Applier(seed)
    assert validate_mod.validate_state(applier.seed_state, invariants=applier.invariants) == []
    assert sf.check_state(applier, applier.seed_state, walker=True) == []


def test_the_generator_reaches_every_operation_and_malformed_edits():
    """Every primitive and macro is generated, and some edits are malformed."""
    import random

    ops: set[str] = set()
    malformed = 0
    for name, seed in sorted(sf.seeds().items()):
        generator = sf.EditGenerator(random.Random(f"coverage:{name}"))
        state = seed.state()
        for _ in range(150):
            edit = generator.edit(state)
            if isinstance(edit, dict) and edit.get("op") in edits_mod.names():
                ops.add(edit["op"])
            else:
                malformed += 1
    assert ops == set(edits_mod.enabled("structure"))
    assert malformed > 0


def test_a_fuzz_run_is_determined_by_its_seed():
    first = sf.Fuzzer(7, walker=False).run(120).summary()
    second = sf.Fuzzer(7, walker=False).run(120).summary()
    assert first == second
    assert first["accepted"] > 0 and first["refused"]


# ── the tiers ─────────────────────────────────────────────────────────────


def test_quick_tier_every_accepted_edit_is_sound():
    stats = sf.Fuzzer(QUICK_SEED, walker=True).run(QUICK_EDITS)
    assert stats.accepted > QUICK_EDITS // 5, _report(stats)
    assert not stats.findings, _report(stats)


@pytest.mark.structure_fuzz_long
@needs_spine
def test_long_tier_every_accepted_edit_is_sound(tmp_path):
    edits = int(os.environ.get("STRUCTURE_FUZZ_EDITS", "4000"))
    seeds = [int(s) for s in os.environ.get("STRUCTURE_FUZZ_SEEDS", "1,2").split(",")]
    totals: list[dict] = []
    findings: list[sf.Finding] = []
    for seed in seeds:
        stats = sf.Fuzzer(seed, walker=True, spine_every=12).run(edits)
        totals.append({"seed": seed, **stats.summary()})
        findings += stats.findings
    report = os.environ.get("STRUCTURE_FUZZ_REPORT")
    if report:
        Path(report).write_text(json.dumps({
            "runs": totals,
            "findings": [{"seed": f.seed, "edits": f.edits, "found": [str(v) for v in f.violations]}
                         for f in findings],
        }, indent=1, default=str))
    assert not findings, json.dumps(totals, indent=1) + "\n" + "\n".join(
        f"{f.seed}: {[str(v) for v in f.violations][:2]} {json.dumps(f.edits, default=str)[:800]}"
        for f in findings[:10])


# ── the regression corpus ─────────────────────────────────────────────────


@pytest.mark.parametrize("case", sf.corpus(), ids=lambda path: path.stem)
def test_regression_corpus_case_replays_clean(case):
    """A case the harness once found: it is refused now, or every state it reaches is sound."""
    found, refusal = sf.replay_case(case)
    assert not found, (f"{case.name} still violates {[str(v) for v in found]}; "
                       f"refused: {refusal}")
    expected = json.loads(case.read_text()).get("refused_with")
    if expected:
        assert refusal and any(reason.startswith(expected) for reason in refusal), refusal


@pytest.mark.structure_fuzz_long
@needs_spine
@pytest.mark.parametrize("case", sf.corpus(), ids=lambda path: path.stem)
def test_regression_corpus_case_replays_clean_on_the_spine_too(case):
    found, _refusal = sf.replay_case(case, spine=True)
    assert not found, [str(v) for v in found]


# ── mutation checks: switch a protection off, and the harness notices ────


def _finds(invariant: str, *, seeds: tuple[str, ...], edits: int = 2500, walker: bool = False,
           spine_every: int = 0, seed: int = 3) -> sf.Stats:
    stats = sf.Fuzzer(seed, seed_names=seeds, walker=walker, spine_every=spine_every,
                      shrink=False, stop_on=invariant).run(edits)
    assert stats.violations[invariant] > 0, _report(stats)
    return stats


def test_mutation_lowering_without_the_fork_rules_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "fork_issues", lambda graph, knobs: [])
    monkeypatch.setattr(base_mod, "_lowering_refusal", lambda parent, state: None)
    _finds("lowering", seeds=("fuzz_fork_join", "fuzz_nested_fork"))


def test_mutation_walker_without_the_data_pass_rules_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "_data_pass_issues", lambda graph, data: [])
    _finds("walker", seeds=("fuzz_dataflow", "fuzz_data_pass", "fuzz_gate_loop"), walker=True,
           edits=4000)


@pytest.mark.structure_fuzz_long
@needs_spine
def test_mutation_spine_without_distinct_gate_verdicts_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "gate_label_issues", lambda graph: [])
    _finds("spine", seeds=("fuzz_gate_loop", "fuzz_multi_terminal"), walker=True, spine_every=1,
           edits=600)


def test_mutation_knobs_without_own_system_prompts_is_caught(monkeypatch):
    monkeypatch.setattr(edits_mod, "_own_bindings", lambda node, graph, state: (node, {}))
    _finds("knobs", seeds=("fuzz_dataflow", "livemath_skill_graph"))


def test_mutation_dataflow_without_the_read_rules_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "read_issues", lambda graph: [])
    monkeypatch.setattr(validate_mod, "agent_input_issues", lambda graph, knobs: [])
    _finds("dataflow", seeds=("fuzz_dataflow", "fuzz_gates"))


def test_mutation_forks_without_the_pairing_rule_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "fork_issues", lambda graph, knobs: [])
    monkeypatch.setattr(base_mod, "_lowering_refusal", lambda parent, state: None)
    _finds("forks", seeds=("fuzz_fork_join", "fuzz_nested_fork"))


def test_mutation_protected_without_the_edit_layer_check_is_caught(monkeypatch):
    monkeypatch.setattr(edits_mod, "protected_violation", lambda before, after, protected: None)
    _finds("protected", seeds=("livemath_skill_graph", "fuzz_data_pass"))


def test_mutation_grader_without_answer_free_items_is_caught(monkeypatch):
    import refactory_dsh.task as task_mod

    monkeypatch.setattr(task_mod, "node_item", lambda item, task, knobs: item)
    _finds("grader", seeds=("fuzz_data_pass", "livemath_skill_graph"), walker=True, edits=200)


def test_mutation_slots_without_the_agent_input_rule_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "agent_input_issues", lambda graph, knobs: [])
    _finds("slots", seeds=("fuzz_dataflow", "fuzz_gates", "fuzz_multi_terminal"), walker=True)


def test_mutation_tools_without_the_new_node_rule_is_caught(monkeypatch):
    monkeypatch.setattr(base_mod, "_new_node_tools", lambda seed, state, allowed: [])
    _finds("tools", seeds=("fuzz_dataflow", "fuzz_gates"))


# ── the rules the harness's findings put in place ─────────────────────────


def test_every_shipped_mode_still_validates():
    from test_lowering_smoke import _mode_classes

    for key, mode_class in _mode_classes():
        mode = mode_class()
        issues = validate_mod.validate_state(mode.state_dict(), invariants=mode.invariants())
        assert issues == [], (key, [str(i) for i in issues])


def test_livemath_items_reach_the_nodes_without_the_answer(tmp_path):
    """The nodes' ``current_item.json`` has no correct choice, label, theorem or sketch;
    the grader still grades with them."""
    seed = sf.seeds()["livemath_skill_graph"]
    state = seed.state()
    result = sf.run_walker(seed, state, tmp_path / "run")
    assert result.exit_ok, result.error
    items = [json.loads(p.read_text()) for p in (tmp_path / "run").rglob("current_item.json")]
    assert items and all(not {"correct_choice", "label", "label_text", "theorem", "sketch"} & set(i)
                         for i in items)
    assert any(r["score"] == 1.0 for r in result.facts["item_scores"])


def test_use_theorem_reveals_the_theorem_through_its_frozen_knob(tmp_path):
    """The theorem reaches the nodes only in a run whose (frozen) use_theorem is on."""
    from refactory_dsh.nn import edits as nn_edits

    seed = sf.seeds()["livemath_skill_graph"]
    state = nn_edits.configure(seed.state(), "use_theorem", True)
    assert state["knobs"]["use_theorem"]["spec"]["frozen"] is True
    model = sf.ScriptedModel(sf._answers(sf._graph_of(state)))
    result = sf.run_walker(seed, state, tmp_path / "run", model)
    assert result.exit_ok, result.error
    items = [json.loads(p.read_text()) for p in (tmp_path / "run").rglob("current_item.json")]
    assert items and all("theorem" in i and "sketch" not in i for i in items)
    assert any("## Theorem" in prompt for _node, prompt, _system in model.log)


def test_node_item_hides_grader_fields_unless_revealed():
    from refactory_dsh.task import Task, node_item

    class Hidden(Task):
        grader_fields = ("answer", "hint")
        revealed_by = {"hint": "show_hint"}

    item = {"id": "1", "question": "q", "answer": "a", "hint": "h"}
    assert node_item(item, Hidden(), {}) == {"id": "1", "question": "q"}
    assert node_item(item, Hidden(), {"show_hint": True}) == {"id": "1", "question": "q", "hint": "h"}
    assert node_item(item, Task(), {}) == item


def test_a_new_agent_runs_without_tools_unless_the_run_allows_them():
    applier = sf.Applier(sf.seeds()["fuzz_dataflow"])
    parent = sf.Candidate.from_state(applier.seed_state)
    node = {"id": "probe", "kind": "agent", "prompt_template": "Look.", "writes": ["probe.txt"]}
    added = applier.attempt(parent, {"op": "insert_after", "target": "load", "node": node}, set())
    assert isinstance(added, sf.Candidate), added
    assert added.state["graph"]["nodes"]["probe"]["tool_policy"] == "none"
    asking = {**node, "tool_policy": None}
    refused = applier.attempt(parent, {"op": "insert_after", "target": "load", "node": asking}, set())
    assert isinstance(refused, list) and refused[0].startswith("E-new-node-tools"), refused
    allowed = base_mod.Optimizer().attempt(
        base_mod.Proposal(edit={"op": "insert_after", "target": "load", "node": asking}),
        parent, applier.invariants, True, seed_state=applier.seed_state, allow_new_node_tools=True)
    assert isinstance(allowed, base_mod.Candidate), allowed


def test_v6_lets_an_op_read_an_optional_file_and_v15_holds_an_agent_to_every_path():
    from refactory_dsh import nn

    def state_with(kind: str) -> dict:
        graph = nn.Graph()
        graph.add(nn.Node("first", "op", call="json:dumps", writes=("a.txt",)))
        reader = (nn.Node("read", "op", call="json:dumps", reads=("b.txt",)) if kind == "op" else
                  nn.Node("read", "agent", prompt_template="Go.", reads=("b.txt",),
                          writes=("r.txt",), tool_policy="none"))
        graph.add(reader)
        graph.add(nn.Node("later", "op", call="json:dumps", writes=("b.txt",)))
        graph.connect("first", "read")
        graph.connect("read", "later")
        return {"graph": graph.to_state(), "knobs": {}}

    assert validate_mod.validate_state(state_with("op")) == []
    codes = {i.code for i in validate_mod.validate_state(state_with("agent"))}
    assert codes == {"V15"}


def test_v15_lets_a_slot_inside_an_if_on_its_name_be_optional():
    from refactory_dsh import nn

    node = nn.Node("a", "agent", prompt_template=(
        "Go.{% if notes %} Notes: {{notes}}{% endif %}{% if other %}{{missing}}{% endif %}"
        "{% if notes == 'x' %}{{notes}}{% else %}{{needed}}{% endif %}"),
        writes=("a.txt",), tool_policy="none")
    assert validate_mod.template_facts(node, {}) == {"missing", "needed"}
    assert sf._needed_slots(node.prompt_template) == {"missing", "needed"}
