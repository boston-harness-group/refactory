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
