"""The structure-edit invariant harness (`structure_fuzz`): tiers, corpus and mutations.

- **Quick tier** (every run): a few hundred seeded random edits over every seed graph,
  through the outer loop's attempt path; every accepted state lowers, runs on the
  walker to a terminal, and keeps the static invariants.
- **Long tier** (``--run-structure-fuzz-long``): thousands of edits per seed, with
  a spine run beside the walker on a sample of the accepted states. Set
  ``STRUCTURE_FUZZ_EDITS`` and ``STRUCTURE_FUZZ_SEEDS`` to change its size, and
  ``STRUCTURE_FUZZ_REPORT`` to a path for its counts as JSON.
- **Compound proposals** (quick tier every run, and in the long tier): walks where some
  steps are lists of 2-4 mixed edits applied as one candidate (``Fuzzer(compound=...)``);
  every accepted combined state keeps every invariant, and each bundle keeps the
  ``compound`` invariant (applied in order, all edits recorded, a refusal naming the
  edit that failed). ``STRUCTURE_FUZZ_COMPOUND`` sets the long tier's share.
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
from typing import Any

import pytest

import structure_fuzz as sf
from refactory_dsh.nn import edits as edits_mod
from refactory_dsh.optim import base as base_mod
from test_livemath_blocks import needs_spine

# `nn` exports the function `validate`, which shadows the module's name on the package.
validate_mod = importlib.import_module("refactory_dsh.nn.validate")

QUICK_SEED = 20261107
QUICK_EDITS = 300
#: The share of compound proposals in the compound tiers.
COMPOUND_SHARE = 0.5


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


def test_the_compound_generator_mixes_every_kind_of_edit():
    """Lists of 2-4 edits, now and then one too many; text edits as values and patches;
    node copies followed by their own prompt knob; structure edits; malformed ones."""
    import random

    sizes: set[int] = set()
    kinds: set[str] = set()
    for name, seed in sorted(sf.seeds().items()):
        generator = sf.EditGenerator(random.Random(f"compound:{name}"))
        state = seed.state()
        for _ in range(120):
            edits = generator.compound(state)["edits"]
            sizes.add(len(edits))
            added: set[str] = set()
            for edit in edits:
                if not isinstance(edit, dict) or edit.get("op") not in edits_mod.names():
                    kinds.add("malformed")
                    continue
                if edit["op"] == "set_knob":
                    kinds.add("patch" if "patch" in edit else "value")
                    if str(edit.get("name")).removesuffix("__system_prompt") in added:
                        kinds.add("new node's prompt knob")
                elif edit["op"] in ("insert_after", "insert_before", "add_node", "wrap"):
                    added.add(str((edit.get("node") or {}).get("id")))
                    kinds.add("structure")
                else:
                    kinds.add("structure")
    assert {2, 3, 4} <= sizes and max(sizes) > base_mod.DEFAULT_MAX_EDITS
    assert kinds == {"malformed", "patch", "value", "new node's prompt knob", "structure"}, kinds


def test_quick_compound_tier_every_accepted_bundle_is_sound():
    stats = sf.Fuzzer(QUICK_SEED, walker=True, compound=COMPOUND_SHARE).run(QUICK_EDITS)
    summary = stats.summary()["compound"]
    assert summary["accepted"] > 20, _report(stats)
    assert summary["refusals_naming_the_edit"] > 0, _report(stats)
    assert "E-too-many-edits" in summary["refused"], _report(stats)
    assert not stats.findings, _report(stats)


@pytest.mark.structure_fuzz_long
@needs_spine
def test_long_tier_every_accepted_edit_is_sound(tmp_path):
    edits = int(os.environ.get("STRUCTURE_FUZZ_EDITS", "4000"))
    seeds = [int(s) for s in os.environ.get("STRUCTURE_FUZZ_SEEDS", "1,2").split(",")]
    share = float(os.environ.get("STRUCTURE_FUZZ_COMPOUND", str(COMPOUND_SHARE)))
    totals: list[dict] = []
    findings: list[sf.Finding] = []
    for seed in seeds:
        stats = sf.Fuzzer(seed, walker=True, spine_every=12).run(edits)
        totals.append({"seed": seed, **stats.summary()})
        findings += stats.findings
        if share:
            stats = sf.Fuzzer(seed, walker=True, spine_every=12, compound=share).run(edits)
            totals.append({"seed": seed, "compound_share": share, **stats.summary()})
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
           spine_every: int = 0, seed: int = 3, compound: float = 0.0) -> sf.Stats:
    stats = sf.Fuzzer(seed, seed_names=seeds, walker=walker, spine_every=spine_every,
                      shrink=False, stop_on=invariant, compound=compound).run(edits)
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
    monkeypatch.setattr(validate_mod, "agent_input_issues", lambda graph, knobs, **_: [])
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
    monkeypatch.setattr(validate_mod, "agent_input_issues", lambda graph, knobs, **_: [])
    _finds("slots", seeds=("fuzz_dataflow", "fuzz_gates", "fuzz_multi_terminal"), walker=True)


def test_mutation_tools_without_the_new_node_rule_is_caught(monkeypatch):
    monkeypatch.setattr(base_mod, "_new_node_tools", lambda seed, state, allowed: [])
    _finds("tools", seeds=("fuzz_dataflow", "fuzz_gates"))


@pytest.mark.skipif(not __import__("refactory_dsh.snippet").snippet.available()[0],
                    reason="no code-node sandbox here: new code nodes are refused before V18")
def test_mutation_code_without_the_static_code_rule_is_caught(monkeypatch):
    """With V18 off, the generator's hostile code gets into accepted states, and the
    ``code`` invariant finds it without running it."""
    monkeypatch.setattr(validate_mod, "code_issues", lambda graph, knobs: [])
    _finds("code", seeds=("fuzz_dataflow", "livemath_skill_graph"))


def test_mutation_grader_without_the_provenance_rule_is_caught(monkeypatch):
    """With V17 off, a node reading what grading produced (under whatever name) gets
    the grade, and the leak invariant finds it by content: the grader-only answers and
    the grader's own verdicts in the finished items' records."""
    monkeypatch.setattr(validate_mod, "grader_issues", lambda *args, **kwargs: [])
    _finds("grader", seeds=("fuzz_data_pass", "fuzz_dataflow"), walker=True, edits=600, seed=1)


def test_the_leak_invariant_reads_what_grading_produced():
    """`grade_secrets` takes the grader's strings from the runtime's per-item records,
    whatever the Task: the synthetic grader's verdict is one of them."""
    import tempfile

    applier = sf.Applier(sf.seeds()["fuzz_data_pass"])
    tmp = Path(tempfile.mkdtemp())
    model = sf.ScriptedModel(sf._answers(sf._graph_of(applier.seed_state)))
    sf.run_walker(applier.seed, applier.seed_state, tmp, model)
    assert any(secret.startswith(sf.fg.VERDICT) for secret in sf.grade_secrets(tmp))


def test_mutation_compound_refusal_without_the_index_is_caught(monkeypatch):
    monkeypatch.setattr(base_mod, "_at", lambda reason, index, steps: reason)
    _finds("compound", seeds=("fuzz_dataflow", "livemath_skill_graph"), edits=600,
           compound=COMPOUND_SHARE)


def test_mutation_compound_applying_only_the_first_edit_is_caught(monkeypatch):
    monkeypatch.setattr(base_mod.Proposal, "steps", property(
        lambda self: (list(self.edits) if self.edits else [self.edit])[:1]))
    _finds("compound", seeds=("fuzz_dataflow", "livemath_skill_graph"), edits=600,
           compound=COMPOUND_SHARE)


def test_mutation_compound_combined_state_unvalidated_is_caught(monkeypatch):
    """Validate only a bundle's first edit's state, not the combined one: the
    invariants on the combined state notice."""
    original = base_mod.Optimizer.attempt

    def first_only(self, proposal, parent, invariants, *args, **kwargs):  # type: ignore[no-untyped-def]
        outcome = original(self, base_mod.Proposal(edit=proposal.steps[0], parent=proposal.parent),
                           parent, invariants, *args, **kwargs)
        if not isinstance(outcome, base_mod.Candidate) or len(proposal.steps) == 1:
            return outcome
        state = outcome.state
        for edit in proposal.steps[1:]:
            try:
                state = edits_mod.apply(edit, state)
            except Exception:  # noqa: BLE001
                return ["E-precondition: mutated"]
        return base_mod.Candidate.from_state(state, parent=parent.id,
                                             edits=[*parent.edits, *proposal.steps],
                                             step_edits=len(proposal.steps))

    monkeypatch.setattr(base_mod.Optimizer, "attempt", first_only)
    stats = sf.Fuzzer(3, seed_names=("fuzz_dataflow", "livemath_skill_graph"), walker=False,
                      shrink=False, compound=COMPOUND_SHARE).run(800)
    spec = {k: v for k, v in stats.violations.items() if k != "compound"}
    assert spec, _report(stats)


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


# ── required reads ───────────────────────────────────────────────────────


def test_mutation_required_without_the_required_read_rule_is_caught(monkeypatch):
    monkeypatch.setattr(validate_mod, "required_read_issues", lambda graph: [])
    _finds("required", seeds=("livemath_skill_graph", "fuzz_dataflow"), walker=True)


@pytest.mark.parametrize("how", ["prompt", "guarded prompt", "system prompt", "feedback_keys"])
def test_v17_refuses_an_agent_that_reads_the_graded_records(how):
    """A data pass's graded records hold the expected answers: an agent after the pass
    that filled them into a prompt was sent the answers (found by the compound long
    tier). However it asks for them, the agent is refused (V17, provenance-based)."""
    applier = sf.Applier(sf.seeds()["fuzz_data_pass"])
    seed = sf.Candidate.from_state(applier.seed_state)
    graded = sorted(n for node in sf._graph_of(seed.state).nodes.values() if node.kind == "data"
                    for n in node.produces)[0]
    node = {"id": "peek", "kind": "agent", "role": "solver", "writes": ["peek.txt"],
            "prompt_template": "Look again.", "system_prompt_template": "You look."}
    if how == "prompt":
        node["prompt_template"] = f"Look at {{{{{graded}}}}}."
    elif how == "guarded prompt":
        node["prompt_template"] = f"{{% if {graded} %}}Look at {{{{ {graded} }}}}.{{% endif %}}"
    elif how == "system prompt":
        node["system_prompt_template"] = f"You know {{{{{graded}}}}}."
    else:
        node["feedback_keys"] = [graded]
    outcome = applier.attempt(seed, {"op": "add_node", "node": node,
                                     "position": {"terminal": True}}, {seed.id})
    assert isinstance(outcome, list) and any(
        r.startswith("V17") and graded in r for r in outcome), outcome


def test_required_reads_are_a_subset_of_an_op_or_gates_reads():
    from refactory_dsh import nn

    with pytest.raises(nn.GraphError, match="not among its reads"):
        nn.Node("a", "op", call="json:dumps", reads=("x",), required_reads=("y",))
    with pytest.raises(nn.GraphError, match="only an op or a gate"):
        nn.Node("a", "agent", prompt_template="p", reads=("x",), required_reads=("x",))
    node = nn.Node("a", "op", call="json:dumps", reads=("x", "y"), required_reads=("x",))
    assert nn.Node.from_state(node.to_state()) == node
    assert "required_reads" not in nn.Node("b", "op", call="json:dumps").to_state()


def _required_state(required: bool) -> dict:
    from refactory_dsh import nn
    from structure_fuzz_graphs import STEP

    graph = nn.Graph()
    graph.add(nn.Node("first", "op", call=STEP, writes=("a.txt",)))
    graph.add(nn.Node("read", "op", call=STEP, reads=("b.txt",),
                      required_reads=("b.txt",) if required else ()))
    graph.add(nn.Node("later", "op", call=STEP, writes=("b.txt",)))
    graph.connect("first", "read")
    graph.connect("read", "later")
    return {"graph": graph.to_state(), "knobs": {}}


def test_v15_holds_a_required_read_to_every_path_and_leaves_an_optional_one():
    assert validate_mod.validate_state(_required_state(False)) == []
    issues = validate_mod.validate_state(_required_state(True))
    assert [i.code for i in issues] == ["V15"] and "required_reads" in issues[0].message


def test_both_runtimes_refuse_a_node_whose_required_read_is_missing(tmp_path):
    from refactory_dsh import lowering
    from refactory_dsh.nn.executor import execute
    from refactory_dsh.run import declarations, missing_inputs

    state = _required_state(True)
    result = execute(sf._graph_of(state), tmp_path / "walk", knobs={})
    assert not result.exit_ok and "required input is missing: b.txt" in str(result.error)
    run = tmp_path / "spine"
    run.mkdir()
    lowering.write_declarations(state, run)
    assert declarations(run, "read")["required_reads"] == ["b.txt"]
    with pytest.raises(ValueError, match="required input is missing: b.txt"):
        missing_inputs(run, declarations(run, "read")["required_reads"])


def test_livemath_ops_require_their_reads_and_the_optimizer_sees_it():
    from refactory_dsh.nn.node_schema import render_schema

    state = sf.seeds()["livemath_skill_graph"].state()
    nodes = state["graph"]["nodes"]
    for op in ("prepare", "notes_prompts", "extract_answer"):
        assert nodes[op]["required_reads"] == nodes[op]["reads"], op
    # decide_prompt goes on with the fork step that answered: only compare's notes are required.
    assert nodes["decide_prompt"]["required_reads"] == ["compare_notes.txt"]
    assert sf._rendered_copy(state, "extract_answer")["required_reads"] == [
        "answer.txt", "current_item.json"]
    assert "required_reads" in "\n".join(render_schema())


# ── split replaces its step; V19 refuses a write nothing reads ───────────


def _split(seed_name: str, target: str, first: dict, second: dict) -> tuple[sf.Applier, Any]:
    applier = sf.Applier(sf.seeds()[seed_name])
    parent = sf.Candidate.from_state(applier.seed_state)
    return applier, applier.attempt(parent, {"op": "split", "target": target,
                                             "into": [first, second]}, {parent.id})


def _agent(node_id: str, prompt: str, writes: str, **extra: Any) -> dict:
    return {"id": node_id, "kind": "agent", "prompt_template": prompt,
            "system_prompt_template": f"You are {node_id}.", "writes": [writes], **extra}


def _writers(graph, name: str) -> list[str]:
    return sorted(n.id for n in graph.nodes.values() if name in n.writes)


def test_split_replaces_its_step_on_the_livemath_graph():
    """decide becomes draft -> final: decide and its own knobs are gone, decide_prompt
    leads to draft, draft to final, final to extract_answer, and answer.txt has one
    writer; the result is sound on the walker (every static invariant too)."""
    applier, out = _split(
        "livemath_skill_graph", "decide",
        _agent("draft", "{{decide_prompt}}", "draft.txt", produces=["draft"]),
        _agent("final", "{{decide_prompt}}\n\nA draft answer: {{draft}}", "answer.txt"))
    assert isinstance(out, sf.Candidate), out
    graph = sf._graph_of(out.state)
    assert "decide" not in graph.nodes
    assert [e.target for e in graph.out_edges("decide_prompt")] == ["draft"]
    assert [e.target for e in graph.out_edges("draft")] == ["final"]
    assert [e.target for e in graph.out_edges("final")] == ["extract_answer"]
    assert not any("decide" in (e.source, e.target) for e in graph.edges)
    assert _writers(graph, "answer.txt") == ["final"]
    assert not [k for k in out.state["knobs"] if k.startswith("decide__")]
    assert {"draft__system_prompt", "draft__task", "final__system_prompt",
            "final__task"} <= set(out.state["knobs"])
    assert sf.check_state(applier, out.state, walker=True) == []


@pytest.mark.parametrize("seed_name, target, writes, expected", [
    # The start node: the first new step becomes the start.
    ("fuzz_dataflow", "load", "raw.txt", {"start": "a", "in": [], "out": ["clean"]}),
    # A data pass's last step: the second new step becomes the pass's exit.
    ("fuzz_data_pass", "solve", "answer.txt", {"in": ["read_item"], "out": ["check"]}),
    # A step a gate reloops to: the reloop edge leads to the first new step.
    ("fuzz_gate_loop", "work", "work.txt", {"in": ["prepare", "review"], "out": ["review"]}),
])
def test_split_rewires_any_graph_shape(seed_name, target, writes, expected):
    applier = sf.Applier(sf.seeds()[seed_name])
    seed_graph = sf._graph_of(applier.seed_state)
    old = seed_graph.nodes[target]
    produces = list(old.produces)
    first = {"id": "a", "kind": "op", "call": old.call or sf.fg.STEP, "writes": ["a.txt"]} \
        if old.kind == "op" else _agent("a", old.prompt_template, "a.txt", produces=["a_fact"])
    second = ({"id": "b", "kind": "op", "call": old.call, "reads": ["a.txt"], "writes": [writes],
               "produces": produces} if old.kind == "op"
              else _agent("b", old.prompt_template + " {{a_fact}}", writes))
    _applier, out = _split(seed_name, target, first, second)
    assert isinstance(out, sf.Candidate), out
    graph = sf._graph_of(out.state)
    assert target not in graph.nodes
    assert sorted(e.source for e in graph.in_edges("a")) == expected["in"]
    assert [e.target for e in graph.out_edges("a")] == ["b"]
    assert [e.target for e in graph.out_edges("b")] == expected["out"]
    if "start" in expected:
        assert graph.start == expected["start"]
    reloop = [e for e in graph.in_edges("a") if e.condition]
    assert all(e.condition == "reloop" for e in reloop)
    assert _writers(graph, writes) == ["b"]
    if seed_name == "fuzz_data_pass":
        assert graph.nodes["items"].subgraph_exit == "check"  # check stays the exit
    assert sf.check_state(applier, out.state, walker=True) == []


@needs_spine
def test_a_split_graph_runs_alike_on_the_walker_and_the_spine():
    applier, out = _split(
        "fuzz_data_pass", "solve",
        _agent("think", "Think about {{question}}", "think.txt", produces=["thoughts"]),
        _agent("answer", "Solve {{question}} using {{thoughts}}", "answer.txt"))
    assert isinstance(out, sf.Candidate), out
    assert sf.spine_eligible(out.state)
    assert sf.check_state(applier, out.state, walker=True, spine=True) == []


@pytest.mark.parametrize("seed_name, target, why", [
    ("fuzz_gates", "check", "is a gate"),
    ("fuzz_fork_join", "split", "is a fork"),
    ("fuzz_fork_join", "left", "a fork's branch or a join's source"),
    ("fuzz_data_pass", "items", "is a data"),
])
def test_split_refuses_a_node_whose_routing_two_steps_cannot_carry(seed_name, target, why):
    _applier, out = _split(seed_name, target, _agent("a", "x", "a.txt", produces=["f"]),
                           _agent("b", "{{f}}", "b.txt"))
    assert isinstance(out, list) and out[0].startswith("E-precondition") and why in out[0], out


def test_split_refuses_to_leave_a_mode_knob_read_by_nothing():
    """depth is read by clean alone: a split of clean must hand it to a new step."""
    first = {"id": "a", "kind": "op", "call": sf.fg.STEP, "reads": ["raw.txt"],
             "writes": ["a.txt"]}
    second = {"id": "b", "kind": "op", "call": sf.fg.STEP, "reads": ["a.txt"],
              "writes": ["clean.txt"], "produces": ["cleaned"]}
    _applier, out = _split("fuzz_dataflow", "clean", first, second)
    assert isinstance(out, list) and "knob(s) ['depth']" in out[0], out
    _applier, out = _split("fuzz_dataflow", "clean", {**first, "knobs": ["depth"]}, second)
    assert isinstance(out, sf.Candidate), out


def test_split_refuses_a_step_whose_own_knob_is_frozen():
    """A frozen knob keeps its value and declaration, so a split that would remove the
    target's own frozen knob is refused up front, saying so."""
    state = sf.seeds()["livemath_skill_graph"].state()
    state["knobs"]["decide__system_prompt"]["spec"]["frozen"] = True
    reason = edits_mod.precheck({"op": "split", "target": "decide", "into": [
        _agent("draft", "{{decide_prompt}}", "draft.txt", produces=["draft"]),
        _agent("final", "{{decide_prompt}} {{draft}}", "answer.txt")]}, state)
    assert reason and "frozen knob(s) ['decide__system_prompt']" in reason, reason


def test_v19_refuses_a_step_whose_output_is_overwritten_before_anyone_reads_it():
    """The audit's case: a new agent writing answer.txt in front of decide, which writes
    it again without reading it, on a straight path. V19 names the lost step."""
    applier = sf.Applier(sf.seeds()["livemath_skill_graph"])
    parent = sf.Candidate.from_state(applier.seed_state)
    early = _agent("early", "{{decide_prompt}}", "answer.txt")
    out = applier.attempt(parent, {"op": "insert_before", "target": "decide", "node": early},
                          {parent.id})
    assert isinstance(out, list) and any(r.startswith("V19 [early]") and "'decide'" in r
                                         for r in out), out


@pytest.mark.parametrize("why, edit", [
    # A reader in between: the op reads the first answer before decide rewrites it.
    ("a reader in between", [
        {"op": "insert_before", "target": "decide", "node": _agent("early", "{{decide_prompt}}",
                                                                    "answer.txt")},
        {"op": "insert_before", "target": "decide", "node": _agent(
            "peek", "Check the first answer.", "peek.txt", reads=["answer.txt"])}]),
    # A fact: the early step's reply is also a fact a later prompt can use.
    ("the reply is a fact", [
        {"op": "insert_before", "target": "decide", "node": _agent(
            "early", "{{decide_prompt}}", "answer.txt", produces=["early_answer"])}]),
])
def test_v19_keeps_a_write_something_can_still_see(why, edit):
    applier = sf.Applier(sf.seeds()["livemath_skill_graph"])
    parent = sf.Candidate.from_state(applier.seed_state)
    out = applier.attempt(parent, {"edits": edit} if len(edit) > 1 else edit[0], {parent.id})
    assert not (isinstance(out, list) and any(r.startswith("V19") for r in out)), (why, out)


_SNIPPET_GATE = "def main(inp):\n    return {'verdict': 'proceed', 'reason': 'ok'}\n"


def _v19(nodes: list, edges: list) -> list[str]:
    """The nodes V19 refuses in a graph built from ``nodes`` and ``edges``."""
    graph = sf.fg._graph(nodes, edges)
    return [issue.node for issue in validate_mod.overwritten_write_issues(graph)]


def _code_gate(node_id: str, reads: tuple[str, ...] = ()) -> Any:
    return sf.nn.Node(node_id, "gate", evaluator="refactory_dsh.snippet:gate",
                      code=_SNIPPET_GATE, reads=reads, max_iterations=2)


def test_v19_leaves_loops_forks_and_ends_alone():
    """A step retried by a gate that reads its file keeps it; one whose gate reads
    something else loses it on every path (reloop: it rewrites it; proceed: the next
    step does); fork branches and the shipped seeds are left alone."""
    for name, seed in sorted(sf.seeds().items()):
        issues = validate_mod.validate_state(seed.state(), invariants=seed.mode().invariants())
        assert not [i for i in issues if i.code == "V19"], name
    edges = [("work", "again"), ("again", "work", "reloop"), ("again", "final", "proceed")]
    work = sf.fg.agent("work", "Do it", writes="work.txt")
    final = sf.fg.agent("final", "Finish", writes="work.txt")
    assert _v19([work, _code_gate("again"), final], edges) == ["work"]
    assert _v19([work, _code_gate("again", reads=("work.txt",)), final], edges) == []


def test_v19_counts_an_agent_with_tools_as_a_possible_reader():
    """An agent with tools may open any file, so a write before it is kept."""
    tools = sf.nn.Node("look", "agent", prompt_template="Look around", writes=("look.txt",))
    assert tools.tool_policy != "none"
    nodes = [sf.fg.agent("first", "One", writes="x.txt"), tools,
             sf.fg.agent("again", "Two", writes="x.txt")]
    assert _v19(nodes, [("first", "look"), ("look", "again")]) == []
    plain = sf.fg.agent("look", "Look", writes="look.txt")
    assert _v19([nodes[0], plain, nodes[2]], [("first", "look"), ("look", "again")]) == ["first"]


def test_v19_walks_a_forks_branches():
    """A file read inside one branch and rewritten after the join is kept."""
    nodes = [
        sf.fg.agent("first", "One", writes="x.txt"),
        sf.nn.Node("fork", "fork", targets=("left", "right")),
        sf.fg.agent("left", "Use it", writes="left.txt", reads=("x.txt",)),
        sf.fg.agent("right", "Other", writes="right.txt"),
        sf.nn.Node("join", "join", sources=("left", "right")),
        sf.fg.agent("again", "Two", writes="x.txt"),
    ]
    edges = [("first", "fork"), ("fork", "join"), ("left", "join"), ("right", "join"),
             ("join", "again")]
    assert _v19(nodes, edges) == []
    nodes[2] = sf.fg.agent("left", "Ignore it", writes="left.txt")
    assert _v19(nodes, edges) == ["first"]


def test_v19_counts_a_data_pass_as_a_possible_reader():
    nodes = [
        sf.fg.agent("first", "One", writes="x.txt"),
        sf.nn.Node("items", "data", subgraph_entry="solve", subgraph_exit="solve",
                   task_ref=sf.fg.TASK_REF, writes=("current_item.json",)),
        sf.fg.agent("solve", "Solve", writes="answer.txt"),
        sf.fg.agent("again", "Two", writes="x.txt"),
    ]
    assert _v19(nodes, [("first", "items"), ("items", "again")]) == []


def test_v19_counts_a_python_op_or_gate_as_a_possible_reader():
    """Neither runtime checks a Python op's reads: one that declares none may still open
    the file, so writer -> op -> rewriter is kept; likewise a Python gate."""
    first, again = (sf.fg.agent("first", "One", writes="x.txt"),
                    sf.fg.agent("again", "Two", writes="x.txt"))
    assert _v19([first, sf.fg.op("mid"), again], [("first", "mid"), ("mid", "again")]) == []
    other = sf.fg.agent("other", "Three", writes="x.txt")
    edges = [("first", "mid"), ("mid", "again", "proceed"), ("mid", "other", "stop")]
    assert _v19([first, sf.fg.gate("mid", "fn"), again, other], edges) == []
    # A code gate reads only what it declares: both verdicts rewrite x.txt unread.
    assert _v19([first, _code_gate("mid"), again, other], edges) == ["first"]


def test_v19_does_not_count_a_code_op_as_a_certain_overwrite():
    """A code op writes only the files its main returns, so it may leave the first write
    in place: the write is kept."""
    code = sf.nn.Node("again", "op", call="refactory_dsh.snippet:run", writes=("x.txt",),
                      code="def main(inp):\n    return {'files': {}}\n")
    assert _v19([sf.fg.agent("first", "One", writes="x.txt"), code], [("first", "again")]) == []
