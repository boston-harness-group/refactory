"""The structure-editing infrastructure every outer loop's proposals go through.

Purpose-free wiring blocks (`nn.operators`: ``collect``, ``select``, who makes each fact
and file, which carry no model output, output placement), the mechanical fixes the
attempt path makes and says (`optim.repair`), the slim repair call, and the checks before
scoring (`optim.checks`: ``E-gate-kind``, the code dry run). The three recorded failure
kinds are regression fixtures here, on neutral graphs: a bundle refused by a split's
precondition although its later edit fixed the reason, a code op that misread its input
file, and a loop gated by a gate nobody answers. Each test fails when the behaviour it
names is reverted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn, snippet
from refactory_dsh.events import render as render_event
from refactory_dsh.loop import Loop
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import operators
from refactory_dsh.nn.executor import execute
from refactory_dsh.optim import AutoresearchOptimizer, Candidate, Optimizer, Proposal, ProposalContext
from refactory_dsh.optim import base, checks, repair

import structure_fuzz as sf
import structure_fuzz_graphs as fg

REPO = Path(__file__).resolve().parents[2]
SPINE = shutil.which("node") is not None and (
    REPO / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists()
needs_spine = pytest.mark.skipif(not SPINE, reason="the spine drain needs Node and the installed "
                                                   "workspace (pnpm install)")
sandbox = pytest.mark.skipif(not snippet.available()[0], reason="no code-node sandbox here")


def _attempt(state: dict[str, Any], edit: Any, *, optimizer: Optimizer | None = None,
             protected: tuple[str, ...] = ()) -> Candidate | list[str]:
    """`Optimizer.attempt` on ``state``, as the outer loop calls it (nobody answers gates)."""
    parent = Candidate.from_state(state)
    proposal = Proposal(edit=edit, parent=parent.id)
    return (optimizer or Optimizer()).attempt(proposal, parent, (), True, (), [parent.id],
                                              seed_state=state, protected=protected)


def _state(seed: str) -> dict[str, Any]:
    return sf.seeds()[seed].state()


# ── the wiring blocks: collect and select, on both runtimes ────────────────


class Wiring(nn.Mode):
    """Two drafts at once, a judge choosing one, ``select`` copying the chosen one, and
    ``collect`` handing a brief, the choice and a file that one route never writes to the
    last step as one fact."""

    name = "wiring_blocks"

    def build(self) -> nn.Graph:
        return fg._graph([
            fg.op("setup", writes=("brief.txt",), produces=("brief",)),
            nn.Node("split", "fork", targets=("draft_a", "draft_b")),
            fg.agent("draft_a", "Draft one from {{brief}}", writes="draft_a.txt"),
            fg.agent("draft_b", "Draft another from {{brief}}", writes="draft_b.txt"),
            nn.Node("merge", "join", sources=("draft_a", "draft_b")),
            fg.agent("judge", "Name the better draft", writes="choice.txt",
                     reads=("draft_a.txt", "draft_b.txt")),
            fg.gate("route", reads=("choice.txt",)),
            fg.op("extra", writes=("extra.txt",)),
            nn.Node("pick", "op", call=operators.SELECT_CALL,
                    reads=("choice.txt", "draft_a.txt", "draft_b.txt"),
                    writes=("chosen.txt",), produces=("chosen",)),
            nn.Node("gather", "op", call=operators.COLLECT_CALL,
                    reads=("brief.txt", "chosen.txt", "extra.txt"), produces=("gathered",)),
            fg.agent("finalize", "Finish from:\n{{gathered}}", writes="answer.txt"),
        ], [("setup", "split"), ("split", "merge"), ("draft_a", "merge"), ("draft_b", "merge"),
            ("merge", "judge"), ("judge", "route"), ("route", "pick", "proceed"),
            ("route", "extra", "reject"), ("extra", "pick"), ("pick", "gather"),
            ("gather", "finalize")])


ANSWERS = {"draft_a": "first draft", "draft_b": "second draft",
           "judge": "draft_b\nIt is clearer.", "finalize": "done"}


def _walk(state: dict[str, Any], workdir: Path) -> nn.RunResult:
    graph = nn.Graph.from_state(state["graph"])
    knobs = {name: entry["value"] for name, entry in (state.get("knobs") or {}).items()}
    return execute(graph, workdir, knobs=knobs, model=sf.ScriptedModel(ANSWERS), max_steps=200)


def test_collect_and_select_wire_files_into_one_fact_on_the_walker(tmp_path):
    state = Wiring().state_dict()
    assert nn.validate_state(state) == [], nn.validate_state(state)
    result = _walk(state, tmp_path)
    assert result.exit_ok, result.error
    chosen = result.facts["chosen"]
    assert chosen == (tmp_path / "draft_b.txt").read_text() and "second draft" in chosen
    assert (tmp_path / "chosen.txt").read_text() == chosen
    gathered = result.facts["gathered"]
    assert gathered.index("### brief.txt\nbrief.txt written by setup") < gathered.index(
        "### chosen.txt\nsecond draft") < gathered.index("### extra.txt\n(not written")


@needs_spine
def test_collect_and_select_agree_on_both_runtimes(tmp_path, monkeypatch):
    """The walker and the spine set the same ``chosen`` and ``gathered`` facts (the
    harness's spine check compares every op's facts)."""
    for node_id, answer in ANSWERS.items():
        monkeypatch.setitem(sf.ANSWERS, node_id, answer)
    seed, state = sf.Seed("wiring_blocks", Wiring), Wiring().state_dict()
    walked, walker = sf.check_walker(seed, state, tmp_path / "walker")
    assert walked == [] and walker is not None and "second draft" in walker.facts["chosen"]
    assert sf.check_spine(seed, state, walker, tmp_path) == []
    spine = json.loads((tmp_path / "spine" / "facts.json").read_text())
    assert spine["gathered"] == walker.facts["gathered"] and "### extra.txt" in spine["gathered"]


class OrderedWiring(nn.Mode):
    """`Wiring` with names whose sorted order is not their declared order: ``select``'s
    label file (``verdict.txt``) sorts after the files it chooses from, and ``collect``
    reads its files in reverse alphabetical order. Both ops read meaning into the order
    their reads were declared in, so a runtime that sorted them would break ``select``
    and reorder ``collect``'s sections."""

    name = "ordered_wiring"

    def build(self) -> nn.Graph:
        return fg._graph([
            fg.op("setup", writes=("brief.txt",), produces=("brief",)),
            nn.Node("split", "fork", targets=("draft_a", "draft_b")),
            fg.agent("draft_a", "Draft one from {{brief}}", writes="a.md"),
            fg.agent("draft_b", "Draft another from {{brief}}", writes="b.md"),
            nn.Node("merge", "join", sources=("draft_a", "draft_b")),
            fg.agent("judge", "Name the better draft", writes="verdict.txt",
                     reads=("a.md", "b.md")),
            nn.Node("pick", "op", call=operators.SELECT_CALL,
                    reads=("verdict.txt", "a.md", "b.md"), writes=("chosen.txt",),
                    produces=("chosen",)),
            nn.Node("gather", "op", call=operators.COLLECT_CALL,
                    reads=("verdict.txt", "chosen.txt", "brief.txt"), produces=("gathered",)),
            fg.agent("finalize", "Finish from:\n{{gathered}}", writes="answer.txt"),
        ], [("setup", "split"), ("split", "merge"), ("draft_a", "merge"), ("draft_b", "merge"),
            ("merge", "judge"), ("judge", "pick"), ("pick", "gather"), ("gather", "finalize")])


ORDERED_ANSWERS = {"draft_a": "first draft", "draft_b": "second draft",
                   "judge": "b\nIt is clearer.", "finalize": "done"}


def _in_declared_order(gathered: str) -> bool:
    return gathered.index("### verdict.txt\n") < gathered.index("### chosen.txt\n") < \
        gathered.index("### brief.txt\n")


def test_select_and_collect_read_their_inputs_in_declared_order_on_the_walker(tmp_path):
    state = OrderedWiring().state_dict()
    assert nn.validate_state(state) == [], nn.validate_state(state)
    graph = nn.Graph.from_state(state["graph"])
    result = execute(graph, tmp_path, model=sf.ScriptedModel(ORDERED_ANSWERS), max_steps=200)
    assert result.exit_ok, result.error
    assert "second draft" in result.facts["chosen"]
    assert _in_declared_order(result.facts["gathered"]), result.facts["gathered"]


@needs_spine
def test_select_and_collect_read_their_inputs_in_declared_order_on_both_runtimes(
        tmp_path, monkeypatch):
    for node_id, answer in ORDERED_ANSWERS.items():
        monkeypatch.setitem(sf.ANSWERS, node_id, answer)
    seed, state = sf.Seed("ordered_wiring", OrderedWiring), OrderedWiring().state_dict()
    walked, walker = sf.check_walker(seed, state, tmp_path / "walker")
    assert walked == [] and walker is not None and "second draft" in walker.facts["chosen"]
    assert sf.check_spine(seed, state, walker, tmp_path) == []
    spine = json.loads((tmp_path / "spine" / "facts.json").read_text())
    assert "second draft" in spine["chosen"] and _in_declared_order(spine["gathered"])


def test_collect_shows_a_long_or_binary_file_as_its_path_and_size(tmp_path, monkeypatch):
    monkeypatch.setattr(operators, "COLLECT_TEXT_CHARS", 10)
    ctx = nn.RunContext(tmp_path)
    ctx.node_id, ctx._declared_reads, ctx._declared_facts = "gather", ("long.txt", "blob.bin"), ("all",)
    (tmp_path / "long.txt").write_text("x" * 50)
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00")
    text = operators.collect(ctx)["all"]
    assert "a file at long.txt, 50 bytes, 50 characters" in text and "x" * 20 not in text
    assert "a binary file at blob.bin, 3 bytes" in text


def test_select_fails_on_a_label_naming_no_file(tmp_path):
    ctx = nn.RunContext(tmp_path)
    ctx.node_id = "pick"
    ctx._declared_reads, ctx._declared = ("choice.txt", "a.txt", "b.txt"), ("out.txt",)
    (tmp_path / "choice.txt").write_text("neither of them")
    with pytest.raises(nn.ExecutionError, match=r"names none of \['a', 'a.txt', 'b', 'b.txt'\]"):
        operators.select(ctx)


# ── who makes each fact and file, and which are the task's own input ───────


def test_task_inputs_are_what_the_model_step_takes_from_plain_ops_without_text_knobs():
    graph = nn.Graph.from_state(_state("fuzz_dataflow")["graph"])
    task = operators.task_inputs(graph, {"depth": 2})
    # load and clean are ops reading only the task's files (clean's knob is a number, not
    # text), and summarise takes in what they make: the task's input as it receives it.
    # emit reads summarise's summary: model output.
    assert task == {("fact", "source"), ("fact", "cleaned"), ("file", "clean.txt")}
    assert not {("file", "summary.txt"), ("fact", "summary"), ("file", "answer.txt")} & task
    assert operators.made_by(graph)[("file", "clean.txt")] == ["clean"]


def test_the_node_inputs_table_says_who_makes_each_one_and_marks_task_inputs():
    state = _state("fuzz_dataflow")
    available = base.available_before(
        state["graph"], {k: e["value"] for k, e in state["knobs"].items()})
    assert available["emit"]["made_by"]["summary.txt"] == ["summarise"]
    assert "clean.txt" in available["summarise"]["task_inputs"]
    assert "summary.txt" not in available["emit"]["task_inputs"]
    text = "\n".join(base._available_section(available))
    # Each written as the pipeline view writes it: one label per fact or file.
    raw = ",".join(available["emit"]["made_by"].get("raw.txt") or ()) or "run"
    assert "summary.txt@summarise" in text and f"raw.txt@{raw}" in text
    assert "(task input)" not in text
    assert "`name@maker`" in text and operators.COLLECT_CALL in text


def test_place_output_puts_a_fact_in_the_next_model_steps_prompt_knob():
    state = Wiring().state_dict()
    graph = nn.Graph.from_state(state["graph"])
    # Bind finalize's task prompt to a knob, as a mode's or an edit's own agents are.
    node = graph.nodes["finalize"]
    graph.nodes["finalize"] = nn.Node.from_state({**node.to_state(), "knobs": ["finalize__task"],
                                                  "field_bindings": [["prompt_template",
                                                                      "finalize__task"]]})
    state = {**state, "graph": graph.to_state(),
             "knobs": {"finalize__task": {"value": node.prompt_template,
                                          "spec": nn.Knob("", "str").spec()}}}
    out, where = operators.place_output(state, "brief", after="pick")
    assert out["knobs"]["finalize__task"]["value"].endswith("## Brief\n{{brief}}")
    assert "finalize" in where and "finalize__task" in where
    with pytest.raises(nn_edits.EditError, match="already fills"):
        operators.place_output(out, "brief", after="pick")
    with pytest.raises(nn_edits.EditError, match="no prompt knob"):
        operators.place_output(state, "brief", after="pick", to="judge")


# ── mechanical fixes the attempt path makes, and says ─────────────────────


def test_a_target_naming_a_steps_file_is_read_as_that_step():
    state = _state("fuzz_dataflow")
    child = _attempt(state, {"op": "insert_after", "target": "clean.txt", "node": fg.op(
        "tidy", reads=("clean.txt",), writes=("tidy.txt",), produces=("tidied",)).to_state()})
    assert isinstance(child, Candidate), child
    assert child.own_edits()[0]["target"] == "clean"
    assert any("'clean.txt' names the file step 'clean' writes" in r for r in child.repairs)
    assert child.describe()["repairs"] == child.repairs


def test_a_taken_id_is_renamed_unless_another_edit_names_it():
    state = _state("fuzz_dataflow")
    node = fg.op("clean", reads=("raw.txt",), writes=("extra.txt",), produces=("extra",)).to_state()
    child = _attempt(state, {"op": "insert_after", "target": "load", "node": node})
    assert isinstance(child, Candidate), child
    assert child.own_edits()[0]["node"]["id"] == "clean2"
    assert any("node id 'clean' is already a step's" in r for r in child.repairs)
    # Another edit of the bundle names the id: which node it means is not clear.
    named = _attempt(state, {"edits": [
        {"op": "insert_after", "target": "load", "node": node},
        {"op": "remove_edge", "source": "clean", "target": "summarise"}]})
    assert isinstance(named, list) and named[0].startswith("E-node"), named
    assert "another edit of the proposal names it" in named[0] and "[edit 1 of 2" in named[0]


def test_a_new_node_with_the_id_of_the_step_its_edit_is_about_is_refused_not_renamed():
    """``insert_after done`` with a node called ``done`` may mean "replace done": it is
    refused with why, never turned into a second step ``done2``."""
    state = _state("fuzz_fork_join")
    node = fg.op("done", reads=("answer.txt",), produces=("finished",)).to_state()
    refused = _attempt(state, {"op": "insert_after", "target": "done", "node": node})
    assert isinstance(refused, list) and refused[0].startswith("E-node"), refused
    assert "is the step this edit is about" in refused[0]
    # Resolved from a file name first, the target is still the step it names.
    refused = _attempt(state, {"op": "insert_before", "target": "answer.txt", "node": fg.agent(
        "decide", "Decide", writes="draft.txt").to_state()})
    assert isinstance(refused, list) and "is the step this edit is about" in refused[0], refused


def test_a_knob_set_to_its_own_value_in_a_bundle_is_dropped_and_said():
    state = _state("fuzz_dataflow")
    child = _attempt(state, {"edits": [{"op": "set_knob", "name": "depth", "value": 2},
                                       {"op": "set_knob", "name": "depth", "value": 4}]})
    assert isinstance(child, Candidate), child
    assert child.own_edits() == [{"op": "set_knob", "name": "depth", "value": 4}]
    assert child.step_edits == 1
    assert any("edit 1 (set_knob) sets knob 'depth' to the value it already holds" in r
               for r in child.repairs)


def test_a_slot_naming_a_file_gets_the_files_whole_text_and_runs_on_both_runtimes():
    """``{{raw}}`` names the file raw.txt: a file_text op before the new step sets the fact
    ``raw`` to the file's whole text, the candidate says so, and the state is sound on the
    walker and (with the workspace installed) on the spine beside it."""
    applier = sf.Applier(sf.seeds()["fuzz_dataflow"])
    applier.optimizer.answered_gates = False
    seed = Candidate.from_state(applier.seed_state)
    critic = fg.agent("critic", "Critique the raw notes:\n{{raw}}", writes="critic.txt").to_state()
    child = applier.attempt(seed, {"op": "insert_after", "target": "clean", "node": critic},
                            {seed.id})
    assert isinstance(child, Candidate), child
    reader = child.state["graph"]["nodes"]["critic_raw_text"]
    assert reader["call"] == operators.FILE_TEXT_CALL and reader["reads"] == ["raw.txt"]
    assert reader["required_reads"] == ["raw.txt"] and reader["produces"] == ["raw"]
    assert child.own_edits()[1]["op"] == "insert_before" and child.step_edits == 2
    assert any("{{raw}} names the file 'raw.txt'" in r and "whole text" in r
               for r in child.repairs)
    assert sf.check_state(applier, child.state, walker=True, spine=SPINE) == []


def test_file_text_hands_a_long_file_whole(tmp_path):
    ctx = nn.RunContext(tmp_path)
    ctx.node_id, ctx._declared_reads, ctx._declared_facts = "text", ("long.txt",), ("long",)
    (tmp_path / "long.txt").write_text("y" * (operators.COLLECT_TEXT_CHARS + 10))
    assert operators.file_text(ctx)["long"] == "y" * (operators.COLLECT_TEXT_CHARS + 10)


def test_a_slot_repair_that_would_rewire_a_protected_step_is_not_made():
    """With ``clean`` protected, a file_text op put before ``summarise`` would rewire
    clean's out-edge (the same insert written by hand is refused E-protected): the slot is
    left for the validator to refuse (V15), not wired."""
    state = _state("fuzz_dataflow")
    value = state["knobs"]["summarise__system_prompt"]["value"] + "\nRaw notes: {{raw}}"
    edit = {"op": "set_knob", "name": "summarise__system_prompt", "value": value}
    refused = _attempt(state, edit, protected=("clean",))
    assert isinstance(refused, list) and refused[0].startswith("V15"), refused
    child = _attempt(state, edit)
    assert isinstance(child, Candidate) and child.repairs, child


def test_fork_branches_and_join_sources_are_not_slot_wired():
    """A step a fork names as a branch (or a join as a source) would not be one any more
    behind a new step: its slot is left for the validator; after the join it is wired."""
    state = _state("fuzz_fork_join")
    graph = nn.Graph.from_state(state["graph"])
    setup = graph.nodes["setup"]
    graph.nodes["setup"] = nn.Node.from_state({**setup.to_state(),
                                               "writes": ["brief.txt", "notes.txt"]})
    state = {**state, "graph": graph.to_state()}

    def with_notes(knob: str) -> dict[str, Any]:
        return {"op": "set_knob", "name": knob,
                "value": state["knobs"][knob]["value"] + "\nNotes: {{notes}}"}

    refused = _attempt(state, with_notes("left__system_prompt"))
    assert isinstance(refused, list) and all(r.startswith("V15") for r in refused), refused
    assert not any("left_notes_text" in r for r in refused), "no step put in front of a branch"
    child = _attempt(state, with_notes("decide__system_prompt"))
    assert isinstance(child, Candidate) and "decide_notes_text" in child.state["graph"]["nodes"]


def test_a_file_two_steps_write_is_not_read_as_either():
    state = _state("fuzz_gates")  # polish and rough both write polished.txt
    refused = _attempt(state, {"op": "insert_after", "target": "polished.txt",
                               "node": fg.op("tidy", produces=("tidied",)).to_state()})
    assert isinstance(refused, list) and refused[0].startswith("E-precondition"), refused


def test_a_slot_with_two_matching_files_is_left_for_the_validator():
    state = _state("fuzz_dataflow")
    graph = nn.Graph.from_state(state["graph"])
    load = graph.nodes["load"]
    graph.nodes["load"] = nn.Node.from_state({**load.to_state(), "writes": ["raw.txt", "raw.md"]})
    state = {**state, "graph": graph.to_state()}
    critic = fg.agent("critic", "Critique {{raw}}", writes="critic.txt").to_state()
    refused = _attempt(state, {"op": "insert_after", "target": "clean", "node": critic})
    assert isinstance(refused, list) and refused[0].startswith("V15"), refused


def test_a_patch_whose_target_is_not_in_the_knob_is_refused_with_why():
    state = _state("fuzz_dataflow")
    child = _attempt(state, {"op": "insert_after", "target": "clean", "node": fg.agent(
        "note", "Note {{cleaned}}", writes="note.txt").to_state()})
    assert isinstance(child, Candidate), child
    patch = {"op": "set_knob", "name": "note__system_prompt",
             "patch": [{"kind": "replace", "target": "not in the text", "content": "x"}]}
    refused = _attempt(child.state, patch)
    assert isinstance(refused, list) and refused[0].startswith("E-patch"), refused
    assert "'not in the text'" in refused[0]
    empty = _attempt(child.state, {**patch, "patch": [{"kind": "replace", "target": "",
                                                       "content": "x"}]})
    assert isinstance(empty, list) and "has no target text" in empty[0], empty
    # A patch that says what to do with a missing target is held to what it says.
    skipped = _attempt(child.state, {**patch, "config": {"on_missing": "skip"}})
    assert isinstance(skipped, list) and skipped[0].startswith("E-noop"), skipped


# ── regression fixtures: the three recorded failure kinds ─────────────────


def test_regression_a_split_bundle_whose_later_edit_removes_the_orphaned_knob_is_accepted():
    """A split that leaves a knob read by nothing, in a bundle whose next edit removes that
    knob, is judged on the bundle's end state (it used to be refused by the split's
    precondition, before the removal ran)."""
    state = _state("fuzz_dataflow")
    split = {"op": "split", "target": "clean", "into": [
        fg.op("trim", reads=("raw.txt",), writes=("trimmed.txt",)).to_state(),
        fg.op("shape", reads=("trimmed.txt",), writes=("clean.txt",),
              produces=("cleaned",)).to_state()]}
    child = _attempt(state, {"edits": [split, {"op": "remove_knob", "name": "depth"}]})
    assert isinstance(child, Candidate), child
    assert "depth" not in child.state["knobs"] and "clean" not in child.state["graph"]["nodes"]
    alone = _attempt(state, split)
    assert isinstance(alone, list) and alone[0].startswith("V7 [depth]"), alone


def _recorded(tmp_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """A parent run directory with one finished item whose current_item.json is ``record``."""
    item = tmp_path / "runs" / "parent" / "passes" / "items" / "one_0"
    item.mkdir(parents=True)
    (item / "current_item.json").write_text(json.dumps(record))
    (item / ".done").write_text("")
    return checks.recorded_inputs(tmp_path / "runs" / "parent")


def _code_op(code: str) -> dict[str, Any]:
    return {"id": "count", "kind": "op", "call": snippet.RUN, "reads": ["current_item.json"],
            "produces": ["part_count"], "code": code}


MISREAD = ("import json\n"
           "def main(inp):\n"
           "    item = json.loads(inp['files']['current_item.json'])\n"
           "    return {'facts': {'part_count': len(item['parts'].items())}}\n")


@sandbox
def test_a_machine_with_no_sandbox_refuses_the_code_dry_run(tmp_path, monkeypatch):
    """No sandbox at all when the dry run runs (`snippet.NoSandbox`, the harness's) still
    refuses the proposal, as before it was told apart from code failures; only a sandbox
    slow to start skips."""
    def no_sandbox(*_: Any, **__: Any) -> Any:
        raise snippet.NoSandbox("code nodes cannot run here: no user namespaces")

    monkeypatch.setattr(snippet, "execute", no_sandbox)
    recorded = _recorded(tmp_path, {"id": "one", "parts": ["a", "b"]})
    optimizer = Optimizer()
    optimizer.recorded_inputs = lambda cid: recorded
    refused = _attempt(_state("fuzz_data_pass"),
                       {"op": "insert_after", "target": "read_item", "node": _code_op(MISREAD)},
                       optimizer=optimizer, protected=("items", "check"))
    assert isinstance(refused, list) and refused[0].startswith("E-code-run"), refused
    assert "code nodes cannot run here: no user namespaces" in refused[0]


@sandbox
def test_regression_a_code_op_that_misreads_its_input_file_is_refused_before_scoring(tmp_path):
    """The new code op treats a list as a mapping: run on the parent's recorded item it
    raises, and the proposal is refused with the error and the line, before any item is
    scored with it."""
    recorded = _recorded(tmp_path, {"id": "one", "parts": ["a", "b"]})
    optimizer = Optimizer()
    optimizer.recorded_inputs = lambda cid: recorded
    refused = _attempt(_state("fuzz_data_pass"),
                       {"op": "insert_after", "target": "read_item", "node": _code_op(MISREAD)},
                       optimizer=optimizer, protected=("items", "check"))
    assert isinstance(refused, list) and refused[0].startswith("E-code-run"), refused
    assert "AttributeError" in refused[0] and "at line 4" in refused[0]
    assert "item 'one_0'" in refused[0]
    fixed = MISREAD.replace("item['parts'].items()", "item['parts']")
    child = _attempt(_state("fuzz_data_pass"),
                     {"op": "insert_after", "target": "read_item", "node": _code_op(fixed)},
                     optimizer=optimizer, protected=("items", "check"))
    assert isinstance(child, Candidate) and child.notes == [], child


@sandbox
def test_the_code_dry_run_is_skipped_with_a_note_when_nothing_was_recorded(tmp_path):
    child = _attempt(_state("fuzz_data_pass"),
                     {"op": "insert_after", "target": "read_item", "node": _code_op(MISREAD)},
                     protected=("items", "check"))
    assert isinstance(child, Candidate), child
    assert child.notes and "code dry run skipped for count" in child.notes[0]
    optimizer = Optimizer()
    optimizer.recorded_inputs = lambda cid: {}
    child = _attempt(_state("fuzz_data_pass"),
                     {"op": "insert_after", "target": "read_item", "node": _code_op(MISREAD)},
                     optimizer=optimizer, protected=("items", "check"))
    assert isinstance(child, Candidate) and "has no recorded inputs" in child.notes[0], child


def _walked_parent(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The data-pass seed run on the walker, as a parent's train run leaves it: its state
    and its recorded inputs (`checks.recorded_inputs`)."""
    seed = sf.seeds()["fuzz_data_pass"]
    state = seed.state()
    run = tmp_path / "runs" / "parent"
    walked, result = sf.check_walker(seed, state, run)
    assert walked == [] and result is not None and result.exit_ok, (walked, result)
    recorded = checks.recorded_inputs(run)
    assert len(recorded["passes"]["items"]) == 2, recorded
    return state, recorded


@sandbox
def test_the_dry_run_uses_a_real_walker_runs_item_files_and_never_its_item_facts(tmp_path):
    """On a parent the walker really ran: a code op misreading an item's file is refused
    with the item's own file; one reading a fact is skipped with a note, since an item
    directory's facts.json is the run's, copied when the pass began (the first item has
    none, the second the first's)."""
    state, recorded = _walked_parent(tmp_path)
    optimizer = Optimizer()
    optimizer.recorded_inputs = lambda cid: recorded
    misread = ("import json\n"
               "def main(inp):\n"
               "    item = json.loads(inp['files']['current_item.json'])\n"
               "    return {'facts': {'part_count': len(item['question'].items())}}\n")
    refused = _attempt(state, {"op": "insert_after", "target": "read_item",
                               "node": _code_op(misread)},
                       optimizer=optimizer, protected=("items", "check"))
    assert isinstance(refused, list) and refused[0].startswith("E-code-run"), refused
    assert "'str' object has no attribute 'items'" in refused[0] and "item 'one_0'" in refused[0]
    uses_fact = {**_code_op("def main(inp):\n    return {'facts': {'part_count': "
                            "len(inp['facts']['question'])}}\n"), "inputs": ["question"]}
    child = _attempt(state, {"op": "insert_after", "target": "read_item", "node": uses_fact},
                     optimizer=optimizer, protected=("items", "check"))
    assert isinstance(child, Candidate), child
    assert "reads the fact(s) question" in child.notes[0], child.notes


@sandbox
@pytest.mark.parametrize("returned", ["{'facts': {'part_count': 1}, 'extra': 1}",
                                      "{'facts': {'undeclared': 1}}",
                                      "{'files': {'not_declared.txt': 'x'}}"])
def test_the_dry_run_refuses_a_result_its_node_cannot_use(tmp_path, returned):
    recorded = _recorded(tmp_path, {"id": "one", "parts": ["a"]})
    optimizer = Optimizer()
    optimizer.recorded_inputs = lambda cid: recorded
    code = f"def main(inp):\n    return {returned}\n"
    refused = _attempt(_state("fuzz_data_pass"),
                       {"op": "insert_after", "target": "read_item", "node": _code_op(code)},
                       optimizer=optimizer, protected=("items", "check"))
    assert isinstance(refused, list) and refused[0].startswith("E-code-run"), refused


def test_gates_the_parent_already_has_are_not_judged_for_who_answers_them():
    """The seed's own person- and agent-answered gates are the author's: an unrelated edit
    on that graph is not refused E-gate-kind."""
    state = _state("fuzz_gates")
    edit = {"op": "set_knob", "name": "redo__system_prompt", "value": "Redo it carefully."}
    assert isinstance(_attempt(state, edit), Candidate)


def test_regression_a_loop_gated_by_a_gate_nobody_answers_is_refused():
    """A loop whose gate an agent answers passes every rule and then fails every item in a
    headless run; it is refused before scoring, and a verdict gate is accepted."""
    state = _state("fuzz_dataflow")
    loop = {"op": "loop", "target": "summarise",
            "gate": {"id": "summary_check", "kind": "gate", "gate_kind": "agent",
                     "max_iterations": 2, "notes": "is the summary complete?"}}
    refused = _attempt(state, loop)
    assert isinstance(refused, list) and refused[0].startswith("E-gate-kind"), refused
    assert "'summary_check' is answered by 'agent'" in refused[0]
    verdict = {**loop, "gate": {"id": "summary_check", "kind": "gate",
                                "evaluator": snippet.VERDICT, "reads": ["summary.txt"],
                                "required_reads": ["summary.txt"], "max_iterations": 2}}
    assert isinstance(_attempt(state, verdict), Candidate)
    # The schema such a run is shown offers neither answered gate kind.
    text = "\n".join(base._node_specs(state["graph"]["nodes"].values(), person_gates=False))
    assert "`gate (agent)`" not in text and "`gate (user)`" not in text


# ── the slim repair call ──────────────────────────────────────────────────


def test_only_refusals_about_mechanics_are_repairable():
    assert repair.repairable(["V15 [x]: needs {{a}}"])
    assert repair.repairable(["E-precondition: no node 'x'", "E-patch: knob 'y'"])
    assert not repair.repairable(["E-noop: leaves the state identical"])
    assert not repair.repairable(["V15 [x]: needs {{a}}", "E-duplicate: exists"])
    assert not repair.repairable(["E-protected: node 'items' is protected"])
    assert not repair.repairable(["V17 [x]: reads grader-derived data"]), "the grader side"
    assert not repair.repairable(["E-too-many-edits: 5 changes"]), "fitting the cap is a choice"
    assert not repair.repairable([])


class _Scripted:
    """A model client answering by the kind of request: a proposal request, or a slim
    repair request; anything else (memory, analysts) gets an empty object."""

    def __init__(self, proposals: list[Any], repairs: list[Any]) -> None:
        self.proposals, self.repairs = list(proposals), list(repairs)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("# Repair a refused proposal"):
            return json.dumps(self.repairs.pop(0) if self.repairs else {"edits": []})
        if prompt.startswith("# Update request"):
            return json.dumps(self.proposals.pop(0) if self.proposals else {"proposals": []})
        return "{}"


def _patch(target: str) -> dict[str, Any]:
    return {"op": "set_knob", "name": "note",
            "patch": [{"kind": "replace", "target": target, "content": "Check twice."}]}


def _repair_run(tmp_path: Path, monkeypatch: Any, client: _Scripted) -> Loop:
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_val_gate import _loop

    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=1, batch_size=1)
        loop.optimizer = AutoresearchOptimizer(client)
        loop.run()
    return loop


def _optimizer_line(loop: Loop) -> dict[str, Any]:
    return next(r for r in map(json.loads, (loop.root / "progress.jsonl").read_text().splitlines())
                if r.get("step") == "optimizer")


def test_a_refused_proposal_is_fixed_by_a_slim_repair_call_that_shows_it(tmp_path, monkeypatch):
    client = _Scripted(
        proposals=[{"proposals": [{"edit": _patch("Be very careful."), "rationale": "say it",
                                   "prediction": "fewer slips"}]}],
        repairs=[{"edits": [_patch("Be careful.")], "why": "quoted the target"}])
    loop = _repair_run(tmp_path, monkeypatch, client)
    full = [p for p in client.prompts if p.startswith("# Update request")]
    slim = [p for p in client.prompts if p.startswith("# Repair a refused proposal")]
    assert len(full) == 1 and len(slim) == 1, [p[:40] for p in client.prompts]
    assert "Be very careful." in slim[0] and "E-patch" in slim[0] and "fewer slips" in slim[0]
    assert "The knobs it edits, as they stand now" in slim[0] and "Be careful." in slim[0]
    # No evidence, history or framing: only what fixing the mechanics needs.
    for section in ("## Evidence from the traces", "## What this run has already tried",
                    "## Recent results", "## What the author says"):
        assert section in full[0] and section not in slim[0], section
    assert len(slim[0]) < 40000
    line = _optimizer_line(loop)
    assert (line["accepted"], line["refused"], line["repaired"], line["repair_calls"]) == (1, 0, 1, 1)
    assert "repaired 1 (E-patch 1) in 1 repair call(s)" in render_event(line)
    [rejection] = loop.rejections
    child = next(c for c in loop.population if c.id == rejection.repaired)
    assert child.state["knobs"]["note"]["value"] == "Check twice."
    assert loop.proposals[child.id]["rationale"] == "say it"
    assert loop.proposals[child.id]["repaired_from"][0].startswith("E-patch")
    trace = [json.loads(line) for line in (loop.root / "optimizer_trace.jsonl").read_text().splitlines()]
    assert [t["kind"] for t in trace if t["kind"].startswith("repair")] == ["repair", "repair_outcome"]
    journal = (loop.root / "journal.jsonl").read_text()
    assert f'"repaired": "{child.id}"' in journal
    # Counted as repaired, not refused, in every tally.
    assert loop._rejection_codes() == {} and loop._repaired_codes() == {"E-patch": 1}
    assert loop.epochs[-1].refused_total == 0
    [line_] = (loop.root / "rejections.jsonl").read_text().splitlines()
    assert json.loads(line_)["repaired"] == child.id
    # The epoch's slim calls are journaled, and a resumed run counts them again.
    assert '"event": "repair_calls"' in journal or '"repair_calls"' in journal
    from test_val_gate import _loop
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    with _patch_mode(TwoLevels):
        again = _loop(tmp_path, epochs=1, batch_size=1)
        again.optimizer = AutoresearchOptimizer(_Scripted([], []))
        again._replay(loop.root / "journal.jsonl")
    assert again.optimizer.repairs_used(line["epoch"]) == 1


def test_when_slim_repairs_fail_the_full_retry_shows_the_refused_proposal(tmp_path, monkeypatch):
    client = _Scripted(
        proposals=[{"proposals": [{"edit": _patch("Be very careful.")}]},
                   {"proposals": [{"edit": _patch("Be careful.")}]}],
        repairs=[{"edits": [_patch("Be quite careful.")]}, {"edits": []}])
    loop = _repair_run(tmp_path, monkeypatch, client)
    kinds = [p.split("\n", 1)[0][:30] for p in client.prompts
             if p.startswith(("# Update request", "# Repair a refused"))]
    assert kinds == ["# Update request — generation ", "# Repair a refused proposal",
                     "# Repair a refused proposal", "# Update request — generation "], kinds
    retry = [p for p in client.prompts if p.startswith("# Update request")][1]
    assert "The latest refused proposal(s) as written" in retry and "Be very careful." in retry
    line = _optimizer_line(loop)
    assert (line["accepted"], line["refused"], line["repaired"], line["repair_calls"]) == (1, 1, 0, 2)
    [rejection] = loop.rejections
    assert any("Be quite careful." in r and "as a slim repair call fixed it" in r
               for r in rejection.reasons), rejection.reasons
    assert "as a slim repair call fixed it" in retry


def test_repair_calls_are_bounded_per_proposal_and_per_epoch():
    calls: list[str] = []

    class Stubborn(Optimizer):
        def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
            calls.append(ctx.epoch)
            return proposal

    optimizer, parent = Stubborn(), Candidate.from_state(_state("fuzz_dataflow"))
    refused = ["V15 [x]: needs {{a}}"]
    again = lambda proposal: list(refused)  # noqa: E731
    ctx = ProposalContext(epoch=1)
    for _ in range(3):
        assert optimizer.repair_refused(ctx, Proposal(edit={"op": "x"}), refused, parent, again) is None
    assert calls == [1, 1, 1], "two for the first proposal, one left for the second"
    assert optimizer.repairs_used(1) == 3
    optimizer.repair_refused(ProposalContext(epoch=2), Proposal(edit={"op": "x"}), refused,
                             parent, again)
    assert calls[-2:] == [2, 2]
    assert optimizer.repair_refused(ctx, Proposal(edit={"op": "x"}), ["E-noop: same"], parent,
                                    again) is None and len(calls) == 5, "an idea refusal: no call"


def test_propose_valid_repairs_through_any_optimizer_that_overrides_repair():
    """The base retry loop (`Optimizer.propose_valid`) sends a refusal to ``repair`` too, and
    records the refusal as repaired into the candidate."""
    state = _state("fuzz_dataflow")

    class Fixer(Optimizer):
        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return [Proposal(edit={"op": "set_knob", "name": "depth", "value": 9})]

        def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
            return Proposal(edit={"op": "set_knob", "name": "depth", "value": 4})

    optimizer = Fixer(k=1)
    accepted, rejections = optimizer.propose_valid(
        ProposalContext(), 1, parents=[Candidate.from_state(state)])
    assert len(accepted) == 1 and accepted[0].state["knobs"]["depth"]["value"] == 4
    assert rejections[0].reasons[0].startswith("E-knob") and rejections[0].repaired == accepted[0].id


def test_a_fixs_refusal_about_the_idea_is_kept_with_the_proposals():
    """When a slim repair's fix is refused for its idea (here a no-op; an integrity veto is
    the same), the refusal is kept with the proposal's and reaches the next request."""
    state = _state("fuzz_dataflow")
    seen: list[ProposalContext] = []

    class NoOpFix(Optimizer):
        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            seen.append(ctx)
            return [Proposal(edit={"op": "set_knob", "name": "depth", "value": 9})]

        def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
            return Proposal(edit={"op": "set_knob", "name": "depth", "value": 2})

    accepted, rejections = NoOpFix(k=2).propose_valid(
        ProposalContext(), 1, parents=[Candidate.from_state(state)])
    assert not accepted and rejections[0].reasons[0].startswith("E-knob")
    assert any(r.startswith("E-noop") and "as a slim repair call fixed it" in r
               for r in rejections[0].reasons), rejections[0].reasons
    assert any(r.startswith("E-noop") for r in seen[1].rejections), seen[1].rejections


def test_a_repaired_proposal_is_held_to_the_rounds_edit_level(tmp_path, monkeypatch):
    """A round asking for a prompt-level change refuses a fix that changes the pipeline
    (``E-level``), as it refuses such a proposal."""
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_val_gate import _loop
    from refactory_dsh.optim.base import Rejection
    from refactory_dsh.outerloop import Run

    class Structural(Optimizer):
        def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
            return Proposal(edit={"op": "insert_after", "target": "work", "node": {
                "id": "extra", "kind": "op", "call": "loop_ops:write_best",
                "writes": ["extra.json"], "produces": ["extra_best"]}})

    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=1, batch_size=1, allowed_ops=())
        loop.optimizer = Structural()
        parent = Candidate.from_state(loop.mode.state_dict())
        loop._levels_wanted, loop._levels_done = ["prompt_knob"], []
        proposal = Proposal(edit=_patch("Be very careful."), parent=parent.id)
        loop.rejections.append(Rejection(1, proposal.edit, ["E-patch: no such target"]))
        assert loop._repair(Run(_loop=loop), proposal, parent, ["E-patch: no such target"]) is None
    assert any("E-level" in r for r in loop.rejections[-1].reasons), loop.rejections[-1].reasons


def test_a_repaired_fix_is_recorded_at_its_own_edit_level(tmp_path, monkeypatch):
    """A round asking for both levels: a prompt-level proposal refused, then fixed into a
    pipeline change, fills the pipeline level, not the prompt level it was written as."""
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    from test_val_gate import _loop
    from refactory_dsh.outerloop import Run

    class Restructure(Optimizer):
        def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
            return Proposal(edit={"op": "insert_after", "target": "finish", "node": {
                "id": "extra", "kind": "op", "call": "loop_ops:write_best",
                "writes": ["extra.json"], "produces": ["extra_best"]}}, parent=parent.id)

    with _patch_mode(TwoLevels):
        loop = _loop(tmp_path, epochs=1, batch_size=1, allowed_ops=())
        loop.optimizer = Restructure()
        parent = Candidate.from_state(loop.mode.state_dict())
        loop._levels_wanted, loop._levels_done = ["prompt_knob", "pipeline"], []
        proposal = Proposal(edit=_patch("Be very careful."), parent=parent.id)
        accepted = loop.install(Run(_loop=loop), [proposal], [parent])
    assert len(accepted) == 1 and "extra" in accepted[0].state["graph"]["nodes"], accepted
    assert loop._levels_done == ["pipeline"], loop._levels_done
    assert loop.rejections[-1].repaired == accepted[0].id


def test_a_rule_is_counted_once_per_proposal_when_a_fixs_refusal_is_kept():
    """The fix's refusal is kept beside the proposal's in the text, but a rule both name
    counts once for that proposal in the run's tallies."""
    from refactory_dsh.loop import _codes
    from refactory_dsh.optim.base import Rejection

    rejection = Rejection(1, {"op": "x"}, [
        "V15 [a]: needs {{b}}", "V15 [a]: needs {{c}} [the proposal as a slim repair call fixed it]",
        "E-noop: same [the proposal as a slim repair call fixed it]"])
    assert len(rejection.reasons) == 3
    assert _codes([rejection]) == {"E-noop": 1, "V15": 1}


# ── what a repair dropped is said ─────────────────────────────────────────

LONG_SYSTEM = "You read every option twice and say which given each one breaks, if any."


def test_a_repair_that_drops_text_says_which_and_keeps_the_rest_quiet():
    original = [{"op": "insert", "after": "clean", "kind": "agent", "role": "judge",
                 "text": "Summarise the cleaned text in three lines.", "system": LONG_SYSTEM}]
    kept = [{**original[0], "role": None}]
    assert repair.lost_content(original, [{k: v for k, v in kept[0].items() if k != "role"}]) == ""
    lost = repair.lost_content(original, [{k: v for k, v in original[0].items()
                                           if k not in ("role", "system")}])
    assert lost == f"insert 'system' ({len(LONG_SYSTEM)} chars)", lost
    changed = repair.lost_content(original, [{**original[0], "text": "Summarise it."}])
    assert changed.startswith("insert 'text'"), changed
    # A renamed id or a moved wiring field is a fix, not lost text.
    assert repair.lost_content([{"op": "insert", "after": "clean", "id": "summary_step_one"}],
                               [{"op": "insert", "after": "load", "id": "summary_step_two"}]) == ""


def test_propose_valid_records_what_a_repair_dropped_and_the_next_request_says_it():
    state = _state("fuzz_dataflow")
    seen: list[ProposalContext] = []
    written = {"op": "set_knob", "name": "depth", "value": 9}

    class Dropper(Optimizer):
        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            seen.append(ctx)
            return [Proposal(edits=(written, {"op": "set_knob", "name": "nonexistent",
                                              "value": LONG_SYSTEM}))]

        def repair(self, ctx, proposal, reasons, parent):  # type: ignore[no-untyped-def]
            return Proposal(edit={"op": "set_knob", "name": "depth", "value": 4})

    accepted, rejections = Dropper(k=2).propose_valid(
        ProposalContext(), 2, parents=[Candidate.from_state(state)])
    assert accepted and rejections[0].repaired == accepted[0].id
    assert rejections[0].dropped == f"edit 2 (set_knob) 'value' ({len(LONG_SYSTEM)} chars)"
    assert seen[1].repair_losses == [{"candidate": accepted[0].id,
                                      "dropped": rejections[0].dropped}]
    [line] = base.repair_loss_lines(seen[1].repair_losses)
    assert accepted[0].id in line and "edit 2 (set_knob) 'value'" in line and len(line) < 300
    assert line in seen[1].render()


def test_the_loop_says_what_a_repair_dropped_in_the_next_request_and_keeps_it(tmp_path,
                                                                               monkeypatch):
    long = "Check twice, then check the units of every quantity again before you answer."
    client = _Scripted(
        proposals=[{"proposals": [{"edit": {"op": "set_knob", "name": "note", "patch": [
            {"kind": "replace", "target": "Be very careful.", "content": long}]}}]}],
        repairs=[{"edits": [_patch("Be careful.")], "why": "quoted the target"}])
    loop = _repair_run(tmp_path, monkeypatch, client)
    slim = next(p for p in client.prompts if p.startswith("# Repair a refused proposal"))
    assert "Keep every text it wrote" in slim and "keep the edit without it" in slim
    [rejection] = loop.rejections
    assert rejection.repaired and rejection.dropped == f"set_knob 'patch' ({len(long)} chars)"
    assert f'"dropped": "{rejection.dropped}"' in (loop.root / "journal.jsonl").read_text()
    request = loop._context().render()
    assert f"`{rejection.repaired}` is your refused proposal as a slim repair fixed it" in request
    # Next to the refusals, not among the trace numbers.
    assert request.index("## Your repaired proposals") < request.index(rejection.dropped)
    assert rejection.dropped not in request[request.index("## Evidence from the traces"):
                                            request.index("## Your repaired proposals")]
    assert rejection.dropped in request
    from test_val_gate import _loop
    from test_compound_proposals import TwoLevels
    from test_skillopt import _patch_mode
    with _patch_mode(TwoLevels):
        again = _loop(tmp_path, epochs=1, batch_size=1)
        again._replay(loop.root / "journal.jsonl")
    assert again.rejections[0].dropped == rejection.dropped
