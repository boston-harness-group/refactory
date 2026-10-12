"""Fidelity expectations come from pinned remote source, never DSH's own graph."""
import json
import hashlib
from pathlib import Path

from benchmarks.create_fidelity.contract import FIXTURES, departures, digest, reference


def test_reference_is_frozen_and_materialized():
    manifest = reference()
    assert manifest["source_commit"] == "3ab6f8f7ca792c874f2654574d7f7fd6dd18b896"
    assert manifest["node_count"] == len(manifest["graph"]["nodes"]) == 29
    assert manifest["edge_count"] == len(manifest["graph"]["edges"]) == 33
    baseline = json.loads((FIXTURES / "baseline.json").read_text())
    assert digest(FIXTURES / "acceptance.py") == baseline["checker_sha256"]
    assert baseline["expected_checker_cases"] == 56
    assert "Acceptance Evidence" in manifest["role_prompts"]["factory/agents/prompts/code_reviewer.md"]


def test_public_baseline_is_not_mistaken_for_fidelity():
    from refactory_dsh.create import CreateMode
    rows = departures(CreateMode().graph())
    assert {r["requirement_id"] for r in rows} >= {"G.start", "G.stages", "G.edges", "G.gate_qa.owner"}


def test_candidate_restores_pinned_contract():
    from refactory_dsh.create_fidelity import CreateFidelityMode
    assert departures(CreateFidelityMode().graph()) == []


def test_public_modes_keep_graph_fingerprints_from_frozen_upstream():
    from refactory_dsh import registry
    frozen = json.loads((FIXTURES / "public-graph-fingerprints.json").read_text())
    package = registry.load("default_pack", Path.cwd())
    for name, expected in frozen["graphs"].items():
        state = package.mode(name)().graph().to_state()
        assert hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest() == expected, name
