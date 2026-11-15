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


# ── who makes each fact and file, and which carry no model output ──────────


def test_model_free_inputs_are_the_tasks_own_and_what_plain_ops_make_from_them():
    graph = nn.Graph.from_state(_state("fuzz_dataflow")["graph"])
    free = operators.model_free_inputs(graph)
    # load and clean are ops reading only the task's files: model-free. summarise is a
    # model step, and emit reads its summary: both carry model output.
    assert {("fact", "source"), ("file", "raw.txt"), ("fact", "cleaned")} <= free
    assert not {("file", "summary.txt"), ("fact", "summary"), ("file", "answer.txt")} & free
    assert operators.made_by(graph)[("file", "clean.txt")] == ["clean"]


def test_the_node_inputs_table_says_who_makes_each_one_and_marks_task_inputs():
    state = _state("fuzz_dataflow")
    available = base.available_before(state["graph"])
    assert available["emit"]["made_by"]["summary.txt"] == ["summarise"]
    assert "raw.txt" in available["emit"]["task_inputs"]
    assert "summary.txt" not in available["emit"]["task_inputs"]
    text = "\n".join(base._available_section(available))
    assert "summary.txt (summarise)" in text and "raw.txt (task input)" in text
    assert "made by" in text and operators.COLLECT_CALL in text


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
    assert isinstance(named, list) and "already exists" in named[0], named


def test_a_knob_set_to_its_own_value_in_a_bundle_is_dropped_and_said():
    state = _state("fuzz_dataflow")
    child = _attempt(state, {"edits": [{"op": "set_knob", "name": "depth", "value": 2},
                                       {"op": "set_knob", "name": "depth", "value": 4}]})
    assert isinstance(child, Candidate), child
    assert child.own_edits() == [{"op": "set_knob", "name": "depth", "value": 4}]
    assert child.step_edits == 1
    assert any("edit 1 (set_knob) sets knob 'depth' to the value it already holds" in r
               for r in child.repairs)


def test_a_slot_naming_a_file_gets_a_collect_op_and_runs_on_both_runtimes():
    """``{{raw}}`` names the file raw.txt: a collect op before the new step hands it the
    file as the fact ``raw``, the candidate says so, and the state is sound on the walker
    and (with the workspace installed) on the spine beside it."""
    applier = sf.Applier(sf.seeds()["fuzz_dataflow"])
    applier.optimizer.answered_gates = False
    seed = Candidate.from_state(applier.seed_state)
    critic = fg.agent("critic", "Critique the raw notes:\n{{raw}}", writes="critic.txt").to_state()
    child = applier.attempt(seed, {"op": "insert_after", "target": "clean", "node": critic},
                            {seed.id})
    assert isinstance(child, Candidate), child
    collector = child.state["graph"]["nodes"]["critic_raw_collect"]
    assert collector["call"] == operators.COLLECT_CALL and collector["reads"] == ["raw.txt"]
    assert collector["produces"] == ["raw"]
    assert child.own_edits()[1]["op"] == "insert_before" and child.step_edits == 2
    assert any("{{raw}} names the file 'raw.txt'" in r for r in child.repairs)
    assert sf.check_state(applier, child.state, walker=True, spine=SPINE) == []


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
