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
    assert "parameter moves only" not in answer
    bundles = json.loads(next(line for line in answer.splitlines()
                              if line.startswith('{"proposals": [{"edits"')))["proposals"]
    assert bundles[0]["edits"][1]["name"] == "<step B's prompt knob>"
    if preset == "conservative":
        # No core operator: the written-out node, and a bundle adding it with its knob.
        assert '"op": "insert_after"' in answer and '"target": "<existing step id>"' in answer
        assert '"name": "<new step id>__system_prompt"' in answer
        assert [b["edits"][0]["op"] for b in bundles] == ["set_knob", "insert_after"]
        return
    # The one-line operator, and a bundle adding a step and telling a later one about it;
    # both bundles, the prompt-knobs one first: neither kind is privileged.
    assert '{"op": "insert", "after": "<existing step id>", "kind": "agent"' in answer
    assert [b["edits"][0]["op"] for b in bundles] == ["set_knob", "insert"]
    assert bundles[1]["edits"][1]["name"] == "<a later step's prompt knob>"


def test_a_prompt_is_patched_by_default_and_a_whole_value_stays_available():
    # SkillOpt edits its skill only by local patches; round 2 (2026-11-14): candidates with
    # a whole-prompt rewrite passed the gate 5/23, patch-only ones 23/46.
    answer = _answer(_render("knob_only", max_edits=4))
    flat = " ".join(answer.split())
    assert "To change a text parameter such as a prompt, patch it, in the form the " \
        "`set_knob` row above gives." in flat
    assert "A whole new `value` is for a setting that is not text, a new step's own prompt" in flat
    assert "neither kind is the safer default" in flat
    bundles = json.loads(next(line for line in answer.splitlines()
                              if line.startswith('{"proposals": [{"edits"')))["proposals"]
    assert all("patch" in e and "value" not in e for e in bundles[0]["edits"])


def test_a_parameters_only_run_shows_no_structural_example():
    text = _render("knob_only", max_edits=4)
    answer = _answer(text)
    assert "parameter moves only" in answer
    # A set_knob patch entry may name the action insert_after; a structural example names it as an op.
    assert not any(op in answer for op in ('"op": "insert_after"', "add_node", "__system_prompt", '"node"'))
    assert "<step A's prompt knob>" in answer, "the bundle example stays two prompt knobs"
    bundles = json.loads(next(line for line in answer.splitlines()
                              if line.startswith('{"proposals": [{"edits"')))["proposals"]
    assert len(bundles) == 1
    assert "## New nodes" not in text and "## Blocks you may insert" not in text


def _examples(answer: str) -> tuple[dict, list[dict]]:
    single = next(json.loads(line.strip()) for line in answer.splitlines()
                  if line.strip().startswith('{"op": "insert'))
    bundle = json.loads(next(line for line in answer.splitlines()
                             if line.startswith('{"proposals": [{"edits"')))
    return single, bundle["proposals"][1]["edits"]


def _fill(value, target: str):  # type: ignore[no-untyped-def]
    text = json.dumps(value)
    for placeholder, filled in (("<existing step id>", target), ("<new step id>", "fresh_step"),
                                ("<prompt text>", "Check the work so far."),
                                ("<what the new step does>", "Check the work so far."),
                                ("<how to use the new step's output>", "Use the check."),
                                ("<system prompt text>", "You check the work.")):
        text = text.replace(placeholder, filled)
    assert "<" not in text, text
    return json.loads(text)


@pytest.mark.parametrize("seed", ["fuzz_draft_review"])
def test_the_examples_parse_and_apply_to_a_generic_graph(seed):
    """Filled in, the structural example and the bundle are accepted by the attempt
    path on a synthetic graph: the node built from the schema is a valid node. (The other
    synthetic graphs' later steps take literal prompts or end the run, so a new step there
    reaches nothing and is refused, E-unread.)"""
    single, bundle = _examples(_answer(_render("broad", max_edits=4)))
    applier = sf.Applier(sf.seeds()[seed])
    parent = Candidate.from_state(applier.seed_state)
    graph = sf._graph_of(parent.state)
    passes = edits._data_pass_nodes(graph)  # noqa: SLF001
    outside = [n for n in sorted(graph.nodes) if n not in passes and graph.nodes[n].kind != "data"]
    target = next((n for n in outside if n not in graph.terminals()), outside[0])
    child = applier.attempt(parent, _fill(single, target), {parent.id})
    assert isinstance(child, Candidate), child
    added = set(child.state["graph"]["nodes"]) - set(parent.state["graph"]["nodes"])
    assert any(f"{n}__task" in child.state["knobs"] for n in added), "its text is a knob of its own"
    # A later step's prompt knob: an agent's system prompt knob downstream of the new step.
    later = next(k for k in sorted(parent.state["knobs"]) if k.endswith("__system_prompt"))
    filled = _fill(json.loads(json.dumps(bundle).replace("<a later step's prompt knob>",
                                                         later)), target)
    bundled = applier.attempt(parent, {"edits": filled}, {parent.id})
    assert isinstance(bundled, Candidate), bundled
    assert bundled.state["knobs"][later]["value"].rstrip().endswith("Use the check.")


def test_new_nodes_come_from_the_schema_without_blocks_and_blocks_are_optional():
    none = _render("broad")
    assert "declares no ready-made blocks" in none and "node schema" in none
    assert "## Blocks you may insert" not in none and "block objects verbatim" not in none
    block = {"id": "b", "kind": "op", "call": "x:y"}
    some = _render("broad", insertable=[block])
    assert "## Blocks you may insert" in some and "or write a node from the node schema" in some
    assert '"call": "x:y"' in some


def test_the_size_cost_is_stated_in_score_units():
    """The 1/N comparison only when the score is known to be the items' mean;
    otherwise the size cost is set against score differences between candidates."""
    records = [{"id": f"i{n}"} for n in range(40)] + [{"id": "i0"}]  # a repeat counts once
    mean = _answer(_render("broad", parsimony=0.02, item_records=records, score_is_item_mean=True))
    assert "0.02 per node (score = raw score - 0.02 x nodes)" in mean
    assert "the raw score is the mean over the 40 train items, so one item is worth 1/40 = 0.025" in mean
    other = _answer(_render("broad", parsimony=0.02, item_records=records))
    assert "1/40" not in other and "score differences between candidates" in other
    assert "Size cost" not in _answer(_render("broad", parsimony=0.0))
    assert "Size cost" not in _answer(_render("broad"))


@pytest.mark.parametrize("mean", [False, True])
def test_the_loop_hands_its_parsimony_and_what_its_score_is(tmp_path, monkeypatch, mean):
    """A run whose every score is the items' mean says one item is worth 1/N; one whose
    score is another metric of the run (here: the mean, shifted) does not."""
    from refactory_dsh.loop import Loop
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        scores = [r["score"] for r in result.item_records]
        result.score = sum(scores) / len(scores) + (0.0 if mean else 0.05)
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, parsimony=0.03, epochs=1)
        loop.run()
        context = loop._context()
    assert context.parsimony == 0.03 and context.score_is_item_mean is mean
    assert ("so one item is worth" in context.render()) is mean


DENYLIST = ("math", "theorem", "option", "LiveMath", "proof", "answer choice")


def test_the_template_names_no_task_or_domain():
    """The request's own words, with no mode data in it, and the reader and optimizer
    prompts: no benchmark or domain vocabulary."""
    texts = [_render("broad", max_edits=4, parsimony=0.01), _render("knob_only", max_edits=4),
             READER_SYSTEM, READER_ANSWER, AutoresearchOptimizer(object())._system,
             base.bundle_instructions(4, edits.catalogue("broad")),
             base.bundle_instructions(4, edits.catalogue("broad"), proposals=1)]
    from refactory_dsh.snippet import ALLOWED_IMPORTS_TEXT

    for text in texts:
        # The code-node import allowlist names the standard library's `math` module, a
        # module name rather than domain vocabulary; it is the one place the word may be.
        text = text.replace(ALLOWED_IMPORTS_TEXT, "")
        for word in DENYLIST:
            assert not re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE), (word, text[:200])


SEPARATE = "separate ideas go in separate proposals"
WHOLE_PATH = "make this one proposal's edits cover every step it names on that\nfailure path (up to 4 changes)"


def test_one_proposal_asks_to_cover_the_whole_failure_path():
    """Asked for one proposal, only one candidate runs: the request asks for one
    proposal covering every step of the diagnosed failure path, not separate ideas
    that would never run. Asked for several (or not said), separate ideas stay
    separate."""
    one = _render("broad", max_edits=4, proposals=1)
    assert WHOLE_PATH in one and "not run this round" in one
    assert SEPARATE not in one
    for text in (_render("broad", max_edits=4, proposals=3), _render("broad", max_edits=4)):
        assert SEPARATE in text and "failure path" not in text
    assert "Bundling" not in _render("broad", max_edits=1, proposals=1)


@pytest.mark.parametrize("n, wording", [(1, WHOLE_PATH), (3, SEPARATE)])
def test_the_optimizer_tells_the_request_how_many_proposals_it_asks_for(n, wording):
    """``AutoresearchOptimizer.propose`` renders the request with its ``n``, so the
    bundling advice matches "Propose at most n update(s)"."""
    class Client:
        prompts: list[str] = []

        def complete(self, prompt, *, system=""):  # type: ignore[no-untyped-def]
            self.prompts.append(prompt)
            return '{"diagnosis": "d", "proposals": []}'

    client = Client()
    AutoresearchOptimizer(client).propose(
        ProposalContext(catalogue=edits.catalogue("broad"), max_edits=4), n, [])
    (prompt,) = client.prompts
    assert wording in prompt and f"Propose at most {n} update(s)" in prompt
    assert (SEPARATE in prompt) is (n > 1)


def test_the_history_shows_edits_compactly_not_whole_values():
    """"What this run has already tried" shows a text set_knob as a length change
    and a bounded diff against the value it replaced (the seed's, then the previous
    edit's), never the whole new value; patches and node objects are bounded too;
    the other columns are unchanged."""
    seed = "\n".join(f"seed line {i} " + "s" * 80 for i in range(60))
    first = seed.replace("seed line 7 ", "FIRST CHANGE ")
    second = first + "\nSECOND ADDITION " + "z" * 3000
    node = {"id": "n2", "kind": "agent", "prompt_template": "P" * 5000}
    rows = [{"id": "c1", "parent": "c0", "score": 0.5, "structural": False, "exit_ok": True,
             "edits": [{"op": "set_knob", "name": "a__system_prompt", "value": first}]},
            {"id": "c2", "parent": "c1", "score": 0.6, "structural": True, "exit_ok": True,
             "edits": [{"op": "set_knob", "name": "a__system_prompt", "value": first},
                       {"op": "set_knob", "name": "a__system_prompt", "value": second},
                       {"op": "insert_after", "target": "a", "node": node},
                       {"op": "set_knob", "name": "b", "patch": [{"find": "x" * 2000}]}]}]
    text = ProposalContext(history=rows, seed_knob_values={"a__system_prompt": seed}).render()
    tried = text[text.index("## What this run has already tried"):].split("\n## ", 1)[0]
    assert first not in tried and "z" * 700 not in tried and "P" * 300 not in tried
    assert "x" * 700 not in tried and "more chars)" in tried
    assert f"set_knob a__system_prompt: {len(seed)} -> {len(first)} chars; diff:" in tried
    assert "-seed line 7 " in tried and "+FIRST CHANGE " in tried
    assert f"set_knob a__system_prompt: {len(first)} -> {len(second)} chars; diff:" in tried
    assert "+SECOND ADDITION" in tried
    assert '"node": {"id": "n2", "kind": "agent", "prompt_template": "PPP' in tried
    assert "set_knob b patch:" in tried
    row = next(line for line in tried.splitlines() if line.startswith("- `c2`"))
    assert row.startswith("- `c2` from `c1`: ") and row.endswith("| train 0.6 (parent 0.5)")
    assert len(tried) < 4000
    unseeded = ProposalContext(history=rows[:1]).render()
    assert f"set_knob a__system_prompt = ({len(first)} chars)" in unseeded
    assert first not in unseeded


def test_the_history_diff_keeps_lines_that_look_like_headers_and_separates_lines():
    """A removed ``---`` line and an added ``++ c`` line are real diff lines, not file
    headers; and the lines stay apart in the table cell."""
    from refactory_dsh.optim.base import DIFF_LINE_SEPARATOR, compact_edits

    old = "a\n---\nb"
    new = "a\nb\n++ c"
    text = compact_edits([{"op": "set_knob", "name": "k", "value": new}], {"k": old})
    assert "----" in text and "+++ c" in text
    assert DIFF_LINE_SEPARATOR + "----" in text and DIFF_LINE_SEPARATOR + "+++ c" in text
    row = ProposalContext(history=[{"id": "c", "edits": [{"op": "set_knob", "name": "k",
                                                          "value": new}]}],
                          seed_knob_values={"k": old}).render()
    assert DIFF_LINE_SEPARATOR + "----" in row


# ── what each node has before it; no person-answered gate without a person ─────


def test_the_request_lists_the_facts_each_node_has_and_how_a_file_becomes_one():
    """On the LiveMath graph, decide has decide_prompt as a fact and compare's notes only
    as a file; the request says a file name is not a fact and a code op makes one."""
    state = sf.seeds()["livemath_skill_graph"].state()
    available = base.available_before(state["graph"])
    assert "decide_prompt" in available["decide"]["facts"]
    assert "compare_notes" not in available["decide"]["facts"]
    assert "compare_notes.txt" in available["decide"]["files"]
    assert "compare_prompt" in available["compare"]["facts"]
    assert "decide_prompt" not in available["compare"]["facts"]
    text = _render(available=available, code_nodes=True)
    # In the reference at the end, with the nodes and edges.
    available_at = "### What each node has before it runs"
    reference = text[text.index(available_at):].split("\n### ", 1)[0]
    rows = {line.split("|")[1].strip(): line for line in reference.splitlines()
            if line.startswith("| ") and not line.startswith("| node(s)")}

    def row_of(node: str) -> str:
        return next(line for nodes, line in rows.items()
                    if node in [n.strip() for n in nodes.split(",")])

    row = row_of("decide")
    assert "decide_prompt@decide_prompt" in row
    # A row that holds another's names says "as at" it: followed back, compare's notes.
    seen = row
    while "compare_notes.txt@compare" not in seen and "as at `" in seen:
        seen = row_of(seen.split("as at `", 1)[1].split("`", 1)[0])
    assert "compare_notes.txt@compare" in seen
    assert "a file's name is not a fact" in reference and "snippet:run" in reference
    plain = _render(available=available, code_nodes=False)
    plain = plain[plain.index(available_at):].split("\n### ", 1)[0]
    assert "a file's name is not a fact" in plain and "snippet:run" not in plain
    assert "facts available" not in _render()


def test_the_schema_offers_no_person_answered_gate_to_a_run_without_a_person(tmp_path):
    """The outer loop's request (its candidates run headless) leaves gate (user) and
    gate (agent) out of the node schemas and says why; with a person attached they are
    offered."""
    from refactory_dsh.loop import Loop
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    nodes = [{"id": "a", "kind": "agent", "prompt_template": "x", "writes": ["a.txt"]}]
    without = _render(node_states=nodes)
    assert "`gate (user)`" not in without and "`gate (fn)`" in without
    assert "`gate (agent)`" not in without
    assert "`gate_kind` `user` and `agent` are not offered" in without
    with_person = _render(node_states=nodes, person_gates=True)
    assert "`gate (user)`" in with_person and "`gate (agent)`" in with_person
    assert "are not offered" not in with_person
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=1)
        loop.init()
        context = loop._context()
    assert context.person_gates is False
    assert "current_item.json" in context.available["work"]["files"]
    assert "`gate (user)`" not in context.render()
    assert isinstance(loop, Loop)


@pytest.mark.parametrize("code_nodes", [True, False])
def test_the_pipeline_ask_names_code_steps_when_the_run_allows_them(code_nodes):
    text = "\n".join(base.levels_instructions(["prompt_knob", "pipeline"], code_nodes=code_nodes))
    assert ("add a code step that checks or extracts" in text) is code_nodes
    assert "loop or parallelise steps" in text


def test_the_available_table_fits_its_limit_on_a_large_graph():
    """A long pipeline's table is capped: cells name fewer facts and say how many more,
    and rows are left out with a note before the section passes its limit. The advice
    names an agent's declared reads, which are pasted into its prompt."""
    from refactory_dsh import nn

    graph = nn.Graph()
    ids = [f"step{n:03d}" for n in range(300)]
    for node_id in ids:
        graph.add(nn.Node(node_id, "op", call="json:dumps", writes=(f"{node_id}_output_file.txt",),
                          produces=(f"{node_id}_fact_with_a_long_name",)))
    for a, b in zip(ids, ids[1:]):
        graph.connect(a, b)
    available = base.available_before(graph.to_state())
    lines = base._available_section(available)
    text = "\n".join(lines)
    assert len(text) <= base.AVAILABLE_SECTION_CHARS
    assert "more)" in text and "not listed" in text
    assert "the files it lists in `reads`, pasted in after the prompt" in lines[0]
    small = "\n".join(base._available_section({"a": {"facts": ["f"], "files": ["x.txt"]}}))
    assert "| a | f@run | x.txt@run |" in small


# ── past edits keep their wiring whole wherever a request shows them ─────────


def _long_split() -> list[dict]:
    """A split whose first step's long prompts come before every step's wiring (the
    LiveMath split the optimizer misread as never reading ``meta_check_notes.txt``)."""
    def step(node_id: str, reads: list[str]) -> dict:
        return {"id": node_id, "kind": "agent",
                "system_prompt_template": f"SYSTEM-{node_id} " + "s" * 3000,
                "prompt_template": f"PROMPT-{node_id} " + "p" * 3000,
                "reads": reads, "writes": [f"{node_id}_notes.txt"],
                "produces": [f"{node_id}_notes"]}
    split = {"op": "split", "target": "decide",
             "into": [step("meta_check", ["current_item.json"]),
                      step("final_decide", ["current_item.json", "meta_check_notes.txt"])]}
    insert = {"op": "insert_after", "target": "final_decide",
              "node": {**step("verify", ["final_decide_notes.txt"]), "max_iterations": 3,
                       "on_exhausted": "continue"}}
    return [split, insert]


def _wiring_whole(text: str, compact: bool = False, whole: bool = False) -> None:
    """Every step's id, reads, writes and produces appear whole; its prompts too with
    ``whole`` (the section is their home), else shortened or pointed to."""
    for node_id, reads in (("meta_check", ["current_item.json"]),
                           ("final_decide", ["current_item.json", "meta_check_notes.txt"]),
                           ("verify", ["final_decide_notes.txt"])):
        assert f'"id": "{node_id}"' in text, node_id
        assert f'"reads": {json.dumps(reads)}' in text, (node_id, reads)
        assert f'"writes": ["{node_id}_notes.txt"]' in text, node_id
        assert f'"produces": ["{node_id}_notes"]' in text, node_id
        assert f"PROMPT-{node_id}" in text, "each prompt keeps its head"
    assert '"max_iterations": 3' in text and '"on_exhausted": "continue"' in text
    if whole:
        assert "p" * 3000 in text and "s" * 3000 in text, "the prompts are whole"
        return
    assert "p" * 700 not in text and ("more chars)" in text or "chars, whole under" in text), \
        "the prompts are shortened, or named where they are whole"


def test_past_structural_edits_keep_every_steps_wiring_in_every_section():
    """'What the last edit changed', the recent results and the history table: each shows
    every step's wiring whole, however long the prompts before it, and shortens only the
    prompts; 'Your earlier reasoning' (and its full rejected edits) points to the first."""
    edits = _long_split()
    row = {"id": "c2", "parent": "c1", "score": 0.5, "structural": True, "exit_ok": True,
           "edits": edits}
    text = ProposalContext(
        edit_pairs=[{"child": "c2", "parent": None, "edit": edits}],
        reasoning=[{"id": "c2", "epoch": 1, "gate": "rejected", "score": 0.5, "fixed": 3,
                    "broken": 1, "edit": edits}],
        leaderboard=[row], history=[row]).render()
    changed = text[text.index("## What the last edit changed"):text.index("## Your earlier")]
    reasoning = text[text.index("## Your earlier"):text.index("### Full edits of")]
    full = text[text.index("### Full edits of"):].split("\n## ", 1)[0]
    recent = text[text.index("## Recent results"):text.index("## Evidence from the traces")]
    tried = text[text.index("## What this run has already tried"):].split("\n## ", 1)[0]
    _wiring_whole(changed, whole=True)
    for section in (recent, tried):
        _wiring_whole(section)
    # The history line names each prompt where it is whole: there, whole.
    assert 'chars, whole under \\"What the last edit changed\\")' in tried
    for node_id in ("meta_check", "final_decide", "verify"):
        assert f"PROMPT-{node_id} " + "p" * 3000 in changed
    # The reasoning and its full rejected edits point to the edit section, which shows
    # them at the full edits' budget, rather than saying the wiring again.
    assert 'under "What the last edit changed" above' in reasoning
    assert 'Its edits are whole under "What the last edit changed" above' in full
    assert '"id"' not in full
    assert len(tried) < 6000, "the history stays bounded"


def test_the_text_budget_shortens_long_texts_and_keeps_short_ones_whole():
    split = _long_split()[0]
    split["into"][0]["prompt_template"] = "short"
    shown = base.fit_edit_text(split, 1000)
    assert shown["into"][0]["prompt_template"] == "short"
    texts = [n[k] for n in shown["into"] for k in ("prompt_template", "system_prompt_template")]
    kept = sum(len(t.split("…", 1)[0]) for t in texts)
    assert kept <= 1000 and all(t.endswith("more chars)") for t in texts[1:])
    assert shown["into"][1]["reads"] == ["current_item.json", "meta_check_notes.txt"]
    long_wire = {"op": "add_edge", "source": "a", "target": "b", "condition": "c" * 900,
                 "note": "n" * 900}
    assert base.fit_edit_text(long_wire, 100) == {**long_wire, "note": "n" * 100
                                                  + "… (800 more chars)"}, "wiring is never cut"


def test_a_patch_entrys_target_is_text_not_wiring():
    """``target`` names a node at an edit's top level, but inside a knob patch it is the
    text being replaced: it is shortened like any text it did not write."""
    patch = {"op": "set_knob", "name": "decide_prompt",
             "patch": [{"target": "T" * 6000, "replacement": "R" * 6000}]}
    text = ProposalContext(
        edit_pairs=[{"child": "c2", "parent": None, "edit": [patch]}],
        reasoning=[{"id": "c2", "epoch": 1, "score": 0.5, "edit": [patch]}],
        history=[{"id": "c2", "parent": "c1", "edits": [patch]}]).render()
    changed = text[text.index("## What the last edit changed"):text.index("## Your earlier")]
    reasoning = text[text.index("## Your earlier"):].split("\n## ", 1)[0]
    tried = text[text.index("## What this run has already tried"):].split("\n## ", 1)[0]
    for section in (changed, reasoning):
        assert "T" * 500 not in section, "the quoted target is shortened"
    # What the patch wrote is whole once, in the child's section; the rest point there.
    assert "R" * 6000 in changed and "R" * 200 not in tried + reasoning
    assert len(tried) < 1500 and len(reasoning) < 1500, "the history keeps its own patch bound"
    assert 'name="decide_prompt"' in changed, "the knob it patches is still named"


def test_a_listing_stays_within_its_text_budget_however_many_steps():
    """Twenty long prompts share one edit's budget even when each one's share is
    tiny: the texts are cut below any floor, down to a bare length marker, and the
    wiring stays whole."""
    into = [{"id": f"s{i}", "kind": "agent", "prompt_template": "p" * 3000,
             "system_prompt_template": "s" * 3000, "reads": [f"s{i - 1}.txt"],
             "writes": [f"s{i}.txt"]} for i in range(10)]
    split = {"op": "split", "target": "decide", "into": into}
    bare = {"op": "split", "target": "decide",
            "into": [{**n, "prompt_template": "", "system_prompt_template": ""} for n in into]}
    for budget in (600, 100):
        shown = base.fit_edit_text(split, budget)
        texts = [n[k] for n in shown["into"] for k in ("prompt_template", "system_prompt_template")]
        assert sum(len(t.split("…", 1)[0]) for t in texts) <= budget, budget
        assert [n["reads"] for n in shown["into"]] == [n["reads"] for n in into]
        assert len(json.dumps(shown)) <= len(json.dumps(bare)) + budget + 25 * len(texts)
    assert all(t.startswith("… (3000 chars)") for n in base.fit_edit_text(split, 100)["into"]
               for t in (n["prompt_template"], n["system_prompt_template"]))
    row = base.compact_edits([split])
    assert len(row) <= len(json.dumps(bare)) + base.HISTORY_EDIT_CHARS + 25 * 20


def test_the_bundle_example_never_shows_more_proposals_than_asked_for():
    """Asked for at most one update, each example answer carries one proposal."""
    import json as _json

    for asked in (1, 2):
        text = base.bundle_instructions(4, edits.catalogue("broad"), proposals=asked)
        shown = [_json.loads(line) for line in text.splitlines() if line.startswith('{"proposals"')]
        assert shown and all(len(answer["proposals"]) <= asked for answer in shown), asked
        assert sum(len(answer["proposals"]) for answer in shown) == 2, "both kinds still shown"
