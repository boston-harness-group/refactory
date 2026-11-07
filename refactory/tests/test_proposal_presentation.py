"""How the proposal request presents structural moves: generically, from the edit
catalogue and the node schema, and only what the run permits.

A run that permits structural ops sees a structural example and a bundle that adds a
node with its own prompt knob; a parameters-only run sees neither. New nodes are said to
come from the node schema when a mode declares no blocks; the size cost is stated with
what it means in score units. No task or domain words appear in the template. Each test
fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
import re

import pytest

from refactory_dsh.nn import edits
from refactory_dsh.optim import AutoresearchOptimizer, Candidate, ProposalContext
from refactory_dsh.optim import base
from refactory_dsh.optim.random_optimizer import READER_ANSWER, READER_SYSTEM

import structure_fuzz as sf


def _answer(text: str) -> str:
    return text[text.index("## Answer"):]


def _render(preset: str = "broad", **fields) -> str:  # type: ignore[no-untyped-def]
    return ProposalContext(catalogue=edits.catalogue(preset), **fields).render()


@pytest.mark.parametrize("preset", ["broad", "structure", "conservative"])
def test_structural_examples_appear_when_structure_is_permitted(preset):
    answer = _answer(_render(preset, max_edits=4))
    assert '"op": "insert_after"' in answer and '"target": "<existing step id>"' in answer
    assert '"name": "<new step id>__system_prompt"' in answer, "the bundle adds a node and its knob"
    assert "parameter moves only" not in answer


def test_a_parameters_only_run_shows_no_structural_example():
    text = _render("knob_only", max_edits=4)
    answer = _answer(text)
    assert "parameter moves only" in answer
    assert not any(op in answer for op in ("insert_after", "add_node", "__system_prompt", '"node"'))
    assert "<step A's prompt knob>" in answer, "the bundle example stays two prompt knobs"
    assert "## New nodes" not in text and "## Blocks you may insert" not in text


def _examples(answer: str) -> tuple[dict, list[dict]]:
    single = next(json.loads(line.strip()) for line in answer.splitlines()
                  if line.strip().startswith('{"op": "insert_after"'))
    bundle = json.loads(next(line for line in answer.splitlines()
                             if line.startswith('{"proposals": [{"edits"')))
    return single, bundle["proposals"][0]["edits"]


def _fill(value, target: str):  # type: ignore[no-untyped-def]
    text = json.dumps(value)
    for placeholder, filled in (("<existing step id>", target), ("<new step id>", "fresh_step"),
                                ("<prompt text>", "Check the work so far."),
                                ("<system prompt text>", "You check the work.")):
        text = text.replace(placeholder, filled)
    assert "<" not in text, text
    return json.loads(text)


@pytest.mark.parametrize("seed", ["fuzz_dataflow", "fuzz_data_pass"])
def test_the_examples_parse_and_apply_to_a_generic_graph(seed):
    """Filled in, the structural example and the bundle are accepted by the attempt
    path on a synthetic graph: the node built from the schema is a valid node."""
    single, bundle = _examples(_answer(_render("broad", max_edits=4)))
    applier = sf.Applier(sf.seeds()[seed])
    parent = Candidate.from_state(applier.seed_state)
    graph = sf._graph_of(parent.state)
    passes = edits._data_pass_nodes(graph)  # noqa: SLF001
    outside = [n for n in sorted(graph.nodes) if n not in passes and graph.nodes[n].kind != "data"]
    target = next((n for n in outside if n not in graph.terminals()), outside[0])
    child = applier.attempt(parent, _fill(single, target), {parent.id})
    assert isinstance(child, Candidate), child
    assert "fresh_step__system_prompt" in child.state["knobs"]
    bundled = applier.attempt(parent, {"edits": _fill(bundle, target)}, {parent.id})
    assert isinstance(bundled, Candidate), bundled
    assert bundled.state["knobs"]["fresh_step__system_prompt"]["value"] == "You check the work."


def test_new_nodes_come_from_the_schema_without_blocks_and_blocks_are_optional():
    none = _render("broad")
    assert "declares no ready-made blocks" in none and "node schema" in none
    assert "## Blocks you may insert" not in none and "block objects verbatim" not in none
    block = {"id": "b", "kind": "op", "call": "x:y"}
    some = _render("broad", insertable=[block])
    assert "## Blocks you may insert" in some and "or write a node from the node schema" in some
    assert '"call": "x:y"' in some


def test_the_size_cost_is_stated_in_score_units():
    records = [{"id": f"i{n}"} for n in range(40)] + [{"id": "i0"}]  # a repeat counts once
    answer = _answer(_render("broad", parsimony=0.02, item_records=records))
    assert "0.02 per node (score = raw score - 0.02 x nodes)" in answer
    assert "one of the 40 train items is worth 1/40 = 0.025 of the raw score" in answer
    assert "Size cost" not in _answer(_render("broad", parsimony=0.0))
    assert "Size cost" not in _answer(_render("broad"))


def test_the_loop_hands_its_parsimony_to_the_request(tmp_path):
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, parsimony=0.03)
        loop.init()
        assert loop._context().parsimony == 0.03


DENYLIST = ("math", "theorem", "option", "LiveMath", "proof", "answer choice")


def test_the_template_names_no_task_or_domain():
    """The request's own words, with no mode data in it, and the reader and optimizer
    prompts: no benchmark or domain vocabulary."""
    texts = [_render("broad", max_edits=4, parsimony=0.01), _render("knob_only", max_edits=4),
             READER_SYSTEM, READER_ANSWER, AutoresearchOptimizer(object())._system,
             base.bundle_instructions(4, edits.catalogue("broad"))]
    for text in texts:
        for word in DENYLIST:
            assert not re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE), (word, text[:200])
