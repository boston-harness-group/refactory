"""Code nodes, verdict gates and ``parallel``: the graph's primitives, composed by proposals.

Every composition here is built the way the outer loop builds a candidate -- a proposal
through ``Optimizer.attempt`` (precheck, apply, protection, validate, lowering) on
LiveMath's skill graph -- then run. Which runtime each test covers is in its name and
docstring: ``walker`` tests run the in-process executor with a scripted model; ``spine``
tests run the real spine drain (`test_livemath_blocks._spine`) and compare its facts with
the walker's through the fuzz harness's ``spine`` invariant.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh import lowering, snippet
from refactory_dsh.nn import RunContext
from refactory_dsh.nn.graph import Graph, GraphError, Node
from refactory_dsh.nn.validate import validate_state
from refactory_dsh.optim import Candidate, Proposal
from test_livemath_blocks import needs_spine

needs_sandbox = pytest.mark.skipif(not snippet.available()[0],
                                   reason=f"no code-node sandbox here: {snippet.available()[1]}")


# ── helpers ───────────────────────────────────────────────────────────────


def _applier() -> sf.Applier:
    return sf.Applier(sf.seeds()["livemath_skill_graph"])


def _propose(edits, applier=None, *, allow_code_nodes=True, secrets=()):
    """The candidate one proposal makes from the seed, or the refusal (a list)."""
    applier = applier or _applier()
    applier.optimizer.allow_code_nodes = allow_code_nodes
    applier.optimizer.grader_secrets = frozenset(secrets)
    parent = Candidate.from_state(applier.seed_state)
    return applier.attempt(parent, {"edits": edits} if len(edits) > 1 else edits[0], {parent.id})


def _accepted(edits, **kwargs):
    outcome = _propose(edits, **kwargs)
    assert not isinstance(outcome, list), outcome
    return outcome.state


class _Model(sf.ScriptedModel):
    """The fuzz harness's scripted model, with per-node answer sequences: a node named in
    ``turns`` answers its list in order (the last one again once it runs out)."""

    def __init__(self, answers, node="", log=None, turns=None, counts=None):
        super().__init__(answers, node, log)
        self.turns = turns if turns is not None else {}
        self.counts = counts if counts is not None else {}

    def for_node(self, node_id, role=""):
        return _Model(self.answers, node_id, self.log, self.turns, self.counts)

    def complete(self, prompt, *, system="", **_):
        self.log.append((self.node, prompt, system))
        if self.node in self.turns:
            index = self.counts.get(self.node, 0)
            self.counts[self.node] = index + 1
            seq = self.turns[self.node]
            return seq[min(index, len(seq) - 1)]
        return self.answers.get(self.node, "<answer>B</answer>")


def _walk(state, tmp_path, *, turns=None):
    model = _Model(sf._answers(sf._graph_of(state)), turns=turns or {})
    result = sf.run_walker(sf.seeds()["livemath_skill_graph"], state, tmp_path / "walker", model)
    return result, model


def _prompts(model, node):
    return [prompt for who, prompt, _system in model.log if who == node]


def _systems(model, node):
    return [system for who, _prompt, system in model.log if who == node]


def _spine_agrees(state):
    """The fuzz harness's checks on one accepted state with the spine on: the walker
    runs it, the spine drain runs it, and both end with the same op facts and scores."""
    applier = _applier()
    assert sf.spine_eligible(state)
    return sf.check_state(applier, state, walker=True, spine=True)


#: A code op after ``compare`` that turns its notes into a fact.
DIGEST = {"op": "insert_after", "target": "compare",
          "node": {"id": "digest", "kind": "op", "call": snippet.RUN,
                   "reads": ["compare_notes.txt"], "required_reads": ["compare_notes.txt"],
                   "produces": ["compare_digest"], "writes": ["digest.txt"],
                   "code": ("import re\n"
                            "def main(inp):\n"
                            "    words = re.findall(r'[a-z]+', inp['files']['compare_notes.txt'])\n"
                            "    return {'facts': {'compare_digest': 'DIGEST:' + '-'.join(words)},\n"
                            "            'files': {'digest.txt': str(len(words))}}\n")}}

#: An agent-verdict loop around ``decide``: decide -> read_answer (code op: the answer
#: file as a fact) -> critic (agent) -> check (verdict gate); reloop -> decide.
CRITIC_LOOP = [
    {"op": "loop", "target": "decide",
     "gate": {"id": "check", "kind": "gate", "evaluator": snippet.VERDICT,
              "reads": ["critic.txt"], "required_reads": ["critic.txt"], "max_iterations": 1}},
    {"op": "insert_before", "target": "check",
     "node": {"id": "read_answer", "kind": "op", "call": snippet.RUN, "reads": ["answer.txt"],
              "required_reads": ["answer.txt"], "produces": ["answer_text"],
              "code": "def main(inp):\n    return {'facts': {'answer_text': inp['files']['answer.txt']}}\n"}},
    {"op": "insert_before", "target": "check",
     "node": {"id": "critic", "kind": "agent", "writes": ["critic.txt"],
              "prompt_template": "Question:\n{{decide_prompt}}\n\nProposed answer:\n{{answer_text}}\n\n"
                                 "Answer 'proceed' or 'reloop' on the first line, then why."}},
]

#: A code-verdict loop around ``decide``: reloop while the answer has no answer tag.
CODE_GATE_LOOP = [
    {"op": "loop", "target": "decide",
     "gate": {"id": "tagged", "kind": "gate", "evaluator": snippet.GATE,
              "reads": ["answer.txt"], "required_reads": ["answer.txt"], "max_iterations": 2,
              "code": ("import re\n"
                       "def main(inp):\n"
                       "    if re.search(r'<answer>[A-Z]</answer>', inp['files']['answer.txt']):\n"
                       "        return {'verdict': 'proceed', 'reason': 'tagged'}\n"
                       "    return {'verdict': 'reloop', 'reason': 'wrap the letter in <answer></answer>'}\n")}},
]

#: Two notes steps at once after ``compare``, joined before ``notes_prompts``.
PARALLEL = {"op": "parallel", "target": "compare", "branches": [
    {"id": "side_a", "kind": "agent", "writes": ["side_a.txt"],
     "prompt_template": "First angle on:\n{{compare_prompt}}"},
    {"id": "side_b", "kind": "op", "call": snippet.RUN, "inputs": ["compare_prompt"],
     "writes": ["side_b.txt"], "produces": ["side_b_len"],
     "code": ("def main(inp):\n    text = inp['facts']['compare_prompt']\n"
              "    return {'facts': {'side_b_len': len(text)}, 'files': {'side_b.txt': text[:50]}}\n")},
]}


# ── the graph fields ──────────────────────────────────────────────────────


def test_code_and_inputs_round_trip_and_only_ops_and_fn_gates_take_them():
    node = Node("c", "op", call=snippet.RUN, code="def main(inp):\n    return {}\n", inputs=("x",))
    assert Node.from_state(node.to_state()) == node
    assert "code" not in Node("p", "op", call="json:dumps").to_state()  # old ids unchanged
    for bad in (dict(kind="agent", prompt_template="hi", code="x"),
                dict(kind="gate", gate_kind="user", inputs=("x",))):
        with pytest.raises(GraphError, match="code or inputs"):
            Node("n", **bad)


# ── the sandbox (the raw sandbox, as V18 would never let these through) ────


@needs_sandbox
@pytest.mark.parametrize("code, refused", [
    ("import socket\ndef main(inp):\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n",
     "OSError"),
    ("def main(inp):\n    return open(inp['facts']['path']).read()\n", "FileNotFoundError"),
    ("import os\ndef main(inp):\n    return os.listdir(inp['facts']['dir'])\n", "ok-empty"),
    ("import os\ndef main(inp):\n    return sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k)\n",
     "ok-empty"),
    ("def main(inp):\n    return len('x' * (2 * 1024 ** 3))\n", "MemoryError"),
])
def test_the_sandbox_denies_network_files_secrets_and_memory(code, refused, tmp_path, monkeypatch):
    """No network; nothing of the run directory or the home directory (each hidden behind
    an empty tmpfs); no credential in the environment; a memory cap."""
    secret_file = tmp_path / "run" / "secret.txt"
    secret_file.parent.mkdir()
    secret_file.write_text("the answer")
    monkeypatch.setenv("SOME_API_KEY", "k")
    inp = {"files": {}, "facts": {"path": str(secret_file), "dir": str(Path.home().parent)}}
    if refused.startswith("ok"):
        assert not snippet._sandboxed(code, inp, timeout_s=10, extra=(str(tmp_path),))
        return
    with pytest.raises(snippet.SnippetError, match=refused):
        snippet._sandboxed(code, inp, timeout_s=10, extra=(str(tmp_path),))


@needs_sandbox
def test_the_sandbox_stops_code_at_its_wall_clock():
    with pytest.raises(snippet.SnippetError, match="over its 2 s limit"):
        snippet._sandboxed("def main(inp):\n    while True:\n        pass\n", {}, timeout_s=2, extra=())


@needs_sandbox
def test_a_code_op_sees_only_its_declared_inputs_and_writes_only_its_declared_outputs(tmp_path):
    """The wrapper hands ``main`` the declared reads and ``inputs`` and nothing else, and
    refuses an output the node does not declare."""
    ctx = RunContext(tmp_path, facts={"wanted": 1, "other": 2}, knobs={"k": "secret knob"})
    (tmp_path / "in.txt").write_text("hello")
    (tmp_path / "undeclared.txt").write_text("hidden")
    ctx.node_id = "c"
    ctx._declared_reads, ctx._declared, ctx._declared_facts = ("in.txt",), ("out.txt",), ("seen",)
    ctx.snippet = {"code": "import json\ndef main(inp):\n    return {'facts': {'seen': json.dumps(inp, "
                           "sort_keys=True)}, 'files': {'out.txt': 'ok'}}\n",
                   "code_knob": "", "inputs": ["wanted"], "required_reads": [], "labels": []}
    seen = json.loads(snippet.run(ctx)["seen"])
    assert seen == {"files": {"in.txt": "hello"}, "facts": {"wanted": 1}}
    assert (tmp_path / "out.txt").read_text() == "ok"
    ctx.snippet = {**ctx.snippet, "code": "def main(inp):\n    return {'facts': {'other': 3}}\n"}
    with pytest.raises(Exception, match="does not declare"):
        snippet.run(ctx)
    ctx.snippet = {**ctx.snippet, "code": "def main(inp):\n    return {'files': {'x.txt': 'y'}}\n"}
    with pytest.raises(Exception, match="does not declare"):
        snippet.run(ctx)


def test_without_the_sandbox_a_code_node_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(snippet, "_probe", lambda: (False, "no namespaces here"))
    with pytest.raises(snippet.SnippetError, match="cannot run here: no namespaces here"):
        snippet.execute("def main(inp):\n    return 1\n", {})
    refused = _propose([DIGEST])
    assert isinstance(refused, list) and refused[0].startswith("E-code-nodes"), refused
    assert "cannot sandbox" in refused[0]


# ── V18 and the request's other rules ─────────────────────────────────────


@pytest.mark.parametrize("marker, code", sf.HOSTILE_STATIC)
def test_v18_refuses_hostile_code_without_running_it(marker, code):
    assert snippet.static_issues(code), marker
    edit = {**DIGEST, "node": {**DIGEST["node"], "code": code}}
    refused = _propose([edit])
    assert isinstance(refused, list) and any(r.startswith("V18") for r in refused), refused


def test_v18_refuses_oversized_code_and_code_on_a_node_that_runs_none():
    big = "def main(inp):\n    return {}\n" + "# pad\n" * 1000
    assert "the most a code node holds" in snippet.static_issues(big)[0]
    graph = Graph.from_nodes([Node("a", "op", call="json:dumps", code="def main(inp):\n    return 1\n")],
                             [], "a")
    issues = validate_state({"graph": graph.to_state(), "knobs": {}})
    assert any(i.code == "V18" and "runs no code" in i.message for i in issues), issues


def test_v18_refuses_a_verdict_gate_with_nothing_to_read():
    edit = {"op": "loop", "target": "decide",
            "gate": {"id": "check", "kind": "gate", "evaluator": snippet.VERDICT}}
    refused = _propose([edit])
    assert any(r.startswith("V18") and "declares neither" in r for r in refused), refused


def test_v17_refuses_a_code_node_whose_inputs_are_grader_derived():
    """``inputs`` count as consumed: a code op after the data node taking its graded
    records as an input is refused by V17 (validated directly: the edit layer's
    protection of ``data`` would refuse the rewiring first)."""
    from refactory_dsh.nn import edits as edits_mod

    applier = _applier()
    edit = {"op": "insert_before", "target": "finish",
            "node": {"id": "peek", "kind": "op", "call": snippet.RUN, "inputs": ["item_scores"],
                     "produces": ["peeked"], "code": "def main(inp):\n    return {}\n"}}
    state = edits_mod.apply(edit, applier.seed_state)
    side = (*applier.protected, *applier.graders)
    issues = validate_state(state, grader_side=side, graders=applier.graders)
    assert any(i.code == "V17" and i.node == "peek" for i in issues), issues
    reader = {**edit, "node": {**edit["node"], "inputs": []}}
    issues = validate_state(edits_mod.apply(reader, applier.seed_state), grader_side=side,
                            graders=applier.graders)
    assert not any(i.code == "V17" for i in issues), issues


def test_v15_holds_inputs_to_every_path():
    edit = {**DIGEST, "node": {**DIGEST["node"], "inputs": ["nowhere_fact"]}}
    refused = _propose([edit])
    assert any(r.startswith("V15") and "nowhere_fact" in r for r in refused), refused


def test_e_secret_refuses_code_that_carries_a_grader_secret():
    secret = "the hidden correct choice text"
    edit = {**DIGEST, "node": {**DIGEST["node"],
                               "code": f"def main(inp):\n    return {{'facts': {{'compare_digest': {secret!r}}}}}\n"}}
    refused = _propose([edit], secrets={secret})
    assert refused and refused[0].startswith("E-secret") and secret not in refused[0], refused
    # The same value set later through the node's code knob is refused too.
    state = _accepted([DIGEST])
    applier = _applier()
    applier.optimizer.grader_secrets = frozenset({secret})
    parent = Candidate.from_state(state)
    again = applier.attempt(parent, {"op": "set_knob", "name": "digest__code",
                                     "value": f"def main(inp):\n    return {{'x': {secret!r}}}\n"},
                            {parent.id})
    assert isinstance(again, list) and again[0].startswith("E-secret"), again


def test_the_run_switch_turns_code_nodes_off():
    refused = _propose([DIGEST], allow_code_nodes=False)
    assert refused and refused[0].startswith("E-code-nodes") and "allow_code_nodes" in refused[0]
    # A verdict gate runs no code, so it is not a code node and stays available.
    critic = {**CRITIC_LOOP[2], "node": {**CRITIC_LOOP[2]["node"],
                                         "prompt_template": "Question:\n{{decide_prompt}}\nproceed or reloop?"}}
    outcome = _propose([CRITIC_LOOP[0], critic], allow_code_nodes=False)
    assert not isinstance(outcome, list), outcome


def test_loop_task_secrets_come_from_the_train_items():
    """The package helper the loop uses (`task.grader_secrets`) finds the LiveMath train
    items' grader-only values, as the fuzz harness's leak check does."""
    with sf.livemath_fixture():
        from refactory_dsh.task import TRAIN, grader_secrets

        secrets = grader_secrets(_applier().seed_state, part=TRAIN)
    assert secrets and all(len(s) >= 8 for s in secrets)


# ── compositions, built by one proposal each, on both runtimes ────────────


@needs_sandbox
def test_walker_code_op_feeds_a_later_step(tmp_path):
    """A code op computes a fact from an agent's file; a later agent's prompt reads it."""
    state = _accepted([DIGEST, {"op": "set_knob", "name": "decide__system_prompt",
                                "patch": [{"kind": "append", "content": "\nNotes digest: {{compare_digest}}"}]}])
    assert state["knobs"]["digest__code"]["value"] == DIGEST["node"]["code"]
    result, model = _walk(state, tmp_path)
    assert result.exit_ok, result.error
    systems = _systems(model, "decide")
    assert systems and all("DIGEST:compare-notes" in s for s in systems), systems[:1]


@needs_sandbox
def test_walker_agent_verdict_gate_relooops_with_the_critics_reason(tmp_path):
    """An agent writes the verdict, a verdict gate routes on it, and the reloop hands the
    critic's reason to the step it redoes; then the critic accepts and the run goes on."""
    state = _accepted(CRITIC_LOOP)
    result, model = _walk(state, tmp_path, turns={
        "critic": ["reloop: show the elimination step", "proceed: fine"]})
    assert result.exit_ok, result.error
    decides = _prompts(model, "decide")
    critics = _prompts(model, "critic")
    items = len(result.facts["item_scores"])
    assert items >= 1
    # The first item: decide, critic (reloop), decide again with the reason, critic (proceed).
    assert "show the elimination step" not in decides[0]
    assert "show the elimination step" in decides[1]
    assert "<answer>B</answer>" in critics[0]  # the critic saw the answer as a fact
    assert len(decides) == items + 1


@needs_sandbox
def test_walker_a_critic_that_never_accepts_is_bounded_by_max_iterations(tmp_path):
    state = _accepted(CRITIC_LOOP)
    result, model = _walk(state, tmp_path, turns={"critic": ["reloop: again"]})
    decides = _prompts(model, "decide")
    critics = _prompts(model, "critic")
    # max_iterations 1: each item is decided at most twice, then the gate halts that pass.
    assert len(decides) == len(critics) and len(decides) % 2 == 0, (len(decides), result.error)


@needs_sandbox
def test_walker_code_verdict_gate_routes_on_computed_checks(tmp_path):
    state = _accepted(CODE_GATE_LOOP)
    assert "tagged__code" in state["knobs"]
    result, model = _walk(state, tmp_path, turns={"decide": ["B, I think", "<answer>B</answer>"]})
    assert result.exit_ok, result.error
    decides = _prompts(model, "decide")
    assert "wrap the letter in <answer></answer>" in decides[1]
    assert all("wrap the letter" not in p for p in decides[2:])


@needs_sandbox
def test_walker_parallel_builds_a_two_branch_fork_in_one_edit(tmp_path):
    state = _accepted([PARALLEL])
    graph = sf._graph_of(state)
    assert graph.nodes["compare_fork"].targets == ("side_a", "side_b")
    assert graph.nodes["compare_join"].sources == ("side_a", "side_b")
    assert "side_a__system_prompt" in state["knobs"] and "side_b__code" in state["knobs"]
    result, model = _walk(state, tmp_path)
    assert result.exit_ok, result.error
    assert _prompts(model, "side_a")
    assert sf.check_static(_applier().seed_state, state, _applier().protected) == []


def test_parallel_refuses_branches_that_would_race():
    clash = {**PARALLEL, "branches": [PARALLEL["branches"][0],
                                      {**PARALLEL["branches"][1], "writes": ["side_a.txt"]}]}
    refused = _propose([clash])
    assert refused[0].startswith("E-precondition") and "both write" in refused[0], refused
    refused = _propose([{**PARALLEL, "branches": PARALLEL["branches"][:1]}])
    assert "2 to 8" in refused[0], refused


def test_lowering_hands_the_spine_the_same_code_node_declaration(tmp_path):
    """Spine: the declarations the runner reads carry the code node's spec as the walker
    builds it, its code read from the state's knob."""
    state = _accepted(CODE_GATE_LOOP + [DIGEST])
    declared = lowering.node_declarations(state)
    graph = sf._graph_of(state)
    knobs = sf._knob_values(state)
    for node_id in ("tagged", "digest"):
        assert declared[node_id]["snippet"] == lowering.snippet_spec(graph, graph.nodes[node_id], knobs)
    assert declared["tagged"]["snippet"]["labels"] == ["proceed", "reloop"]
    lowering.to_workflow(state)


def test_spine_runner_prints_a_verdict_gates_reason_then_its_label(tmp_path, monkeypatch, capsys):
    """Spine: the node runner (`python -m refactory_dsh.run`) runs the verdict gate from
    the lowered declarations and prints the reason, then the label -- the spine routes on
    the last line and hands the lines before it to the step a reloop goes back to."""
    from refactory_dsh import run as run_mod

    state = _accepted(CRITIC_LOOP)
    lowering.write_declarations(state, tmp_path)
    (tmp_path / "critic.txt").write_text("**RELOOP** - show the elimination step\nand check B")
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "check", "type": "gate"}))
    monkeypatch.setenv("REFACTORY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    assert run_mod.main([snippet.VERDICT]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "reloop"
    assert "\n".join(lines[:-1]) == "show the elimination step\nand check B"
    (tmp_path / "critic.txt").write_text("I am not sure")
    assert run_mod.main([snippet.VERDICT]) == 1


@needs_sandbox
@needs_spine
@pytest.mark.parametrize("name, edits", [
    ("code op", [DIGEST]),
    ("agent verdict gate", CRITIC_LOOP),
    ("code verdict gate", CODE_GATE_LOOP),
    ("parallel", [PARALLEL]),
])
def test_spine_runs_each_composition_as_the_walker_does(name, edits, monkeypatch):
    """Spine and walker: the fuzz harness's invariants with the spine on -- same op facts
    (a code op's included) and item scores on both runtimes, no leak, every terminal.
    The critic answers 'proceed' on both (the drain gives each agent one answer)."""
    monkeypatch.setitem(sf.ANSWERS, "critic", "proceed: the answer follows from the notes")
    state = _accepted(edits)
    assert _spine_agrees(state) == []


# ── the request ───────────────────────────────────────────────────────────


def test_the_request_describes_the_primitives_with_examples_from_this_graph():
    from refactory_dsh.nn import edits as edits_mod
    from refactory_dsh.optim.base import composition_section

    applier = _applier()
    nodes = [dict(state, id=node_id) for node_id, state in applier.seed_state["graph"]["nodes"].items()]
    edges = applier.seed_state["graph"]["edges"]
    on = "\n".join(composition_section(edits_mod.catalogue(), nodes, edges,
                                       protected=applier.protected, code_nodes=True))
    off = "\n".join(composition_section(edits_mod.catalogue(), nodes, edges,
                                        protected=applier.protected, code_nodes=False))
    for text in (on, off):
        assert snippet.VERDICT in text and "parallel" in text and "max_iterations" in text
        assert '"target": "compare"' in text  # an example built from this graph's ids
    assert snippet.RUN in on and snippet.RUN not in off and snippet.GATE not in off
    assert composition_section([{"op": "set_knob", "kind": "knob", "fields": []}], nodes, edges) == []
    # Every example in it is a proposal this graph accepts.
    for line in on.splitlines():
        if line.startswith('  {"'):
            example = json.loads(line)
            outcome = _propose(example.get("edits") or [example], applier=_applier())
            assert not isinstance(outcome, list) or not any(
                r.startswith(("E-op", "E-shape", "V1", "V2", "V3", "V4", "V5", "V10", "V12", "V16", "V18"))
                for r in outcome), (example, outcome)
