"""The walker's prompts and run files, as the spine's (dm-123, #436 gaps 5, 13, 14, 18, 20).

`test_runtime_equivalence.py` compares one design run on both runtimes. These hold the
same rules on small graphs of any shape: a node's prompt is its rendered template with a
reloop's feedback said once and its inputs in name order, never its notes; the run leaves
`facts.json` and `.refactory/spine-status.json` as the spine does, with the nodes
completed by the spine's rules; a gate's verdict is the run's record, not a fact; and the
readers that ask whether a walked run is still alive keep asking its own progress.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from refactory_dsh import nn
from refactory_dsh.nn.executor import PROGRESS


class FakeModel:
    """A model client that records what it was asked."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, system: str = "") -> str:
        self.prompts.append((system, prompt))
        return "ok"


def _reloop_to_agent(gate: str, template: str) -> nn.Graph:
    """An agent, a gate that sends it back twice and says why, and an op after."""
    graph = nn.Graph()
    graph.add(nn.Node("write", "agent", prompt_template=template, writes=("answer.txt",),
                      notes="write the answer"))
    graph.add(nn.Node("gate", "gate", evaluator=gate, max_iterations=4, produces=("passes", "why")))
    graph.add(nn.Node("done", "op", call="loop_ops:noop"))
    graph.connect("write", "gate")
    graph.connect("gate", "write", condition="retry")
    graph.connect("gate", "done", condition="proceed")
    return graph


def _status(run) -> dict:
    return json.loads((run / ".refactory/spine-status.json").read_text())


# ── the prompt ────────────────────────────────────────────────────────────────


def test_a_nodes_notes_are_not_part_of_its_prompt(tmp_path):
    """The spine's worker sends the rendered template alone; so does the walker."""
    model = FakeModel()
    result = nn.execute(_reloop_to_agent("loop_ops:three_passes", "Write it."), tmp_path, model=model)
    assert result.exit_ok, result.error
    assert [system for system, _ in model.prompts] == ["", "", ""]
    assert not any("write the answer" in prompt for _, prompt in model.prompts)


def test_a_reloop_reason_reaches_the_node_it_sends_back_once(tmp_path):
    """A template that does not carry the gate's reason gets it under the feedback heading,
    for the one visit it was given for, as the spine's worker shows it."""
    model = FakeModel()
    result = nn.execute(_reloop_to_agent("loop_ops:explained_passes", "Write it."), tmp_path, model=model)
    assert result.exit_ok, result.error
    first, second, third = (prompt for _, prompt in model.prompts)
    assert "## Feedback for this visit" not in first
    assert second.count("## Feedback for this visit\n\npass 1 of 3\n\nWrite it.") == 1
    assert "pass 2 of 3" in third and "pass 1 of 3" not in third


def test_feedback_is_for_the_visit_it_was_given_for(tmp_path):
    """Spent once the node ran: the next item's pass of the same node does not repeat the
    reason the last item's gate gave (`Spine.completeDue` drops an agent's feedback)."""
    from refactory_dsh.nn.executor import CURRENT_ITEM

    graph = nn.Graph()
    graph.add(nn.Node("items", "data", subgraph_entry="write", subgraph_exit="next",
                      writes=(CURRENT_ITEM,), inline_items=({"id": "a"}, {"id": "b"})))
    graph.add(nn.Node("write", "agent", prompt_template="Write it.", reads=(CURRENT_ITEM,), writes=("answer.txt",)))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:retry_each_item_once", max_iterations=4,
                      writes=("tried.json",)))
    graph.add(nn.Node("next", "op", call="loop_ops:noop"))
    graph.connect("write", "gate")
    graph.connect("gate", "write", condition="retry")
    graph.connect("gate", "next", condition="proceed")
    model = FakeModel()
    result = nn.execute(graph, tmp_path, model=model)
    assert result.exit_ok, result.error
    prompts = [prompt for _, prompt in model.prompts]
    assert ["item a" in prompt for prompt in prompts] == [False, True, False, False]
    assert ["item b" in prompt for prompt in prompts] == [False, False, False, True]


def test_a_reloop_reason_the_template_renders_is_not_repeated(tmp_path):
    """A template that renders the reason already says it: no second copy (gap 5)."""
    model = FakeModel()
    result = nn.execute(_reloop_to_agent("loop_ops:explained_passes_in_a_fact", "Write it.\nLast time:\n{{why}}"),
                        tmp_path, model=model)
    assert result.exit_ok, result.error
    second = model.prompts[1][1]
    assert "## Feedback for this visit" not in second
    assert second.count("pass 1 of 3") == 1


def test_feedback_counts_as_said_only_as_whole_lines():
    """A note the prompt contains only inside a longer line is not said: "add tests" is not
    "add tests for every change", nor "fails" "The test suite fails often"."""
    from refactory_dsh.agent_delivery import FEEDBACK_HEADING, with_feedback

    for prompt, note in (("Do it.\nPlease add tests for every change.", "add tests"),
                         ("The test suite fails often.", "fails"),
                         ("first line\nsecond", "first line\nsecond line"),
                         # Every line is there, but not in a row: not the same words.
                         ("add tests\nunrelated\nfor the widget", "add tests\nfor the widget")):
        assert with_feedback(prompt, note) == f"{FEEDBACK_HEADING}\n\n{note}\n\n{prompt}", (prompt, note)
    said = "Do it.\n## What went wrong last time\nadd tests\nfor the widget\n\nGo."
    assert with_feedback(said, "add tests\nfor the widget") == said


def test_a_reason_past_the_runners_limit_the_template_renders_is_said_once(tmp_path):
    """Clipped to the runner's limit (head, marker, tail), a long check log is still the
    one the template rendered in full: it is not prepended a second time."""
    from loop_ops import LONG_REASON

    model = FakeModel()
    result = nn.execute(_reloop_to_agent("loop_ops:long_reason_once", "Write it.\n\n## Last time\n{{why}}\n"),
                        tmp_path, model=model)
    assert result.exit_ok, result.error
    second = model.prompts[1][1]
    assert "## Feedback for this visit" not in second and "clipped from" not in second
    assert second.count("check log line 149:") == 1 and len(LONG_REASON) > 6000


def test_a_reason_past_the_runners_limit_reaches_the_node_clipped_as_the_spine_clips_it(tmp_path):
    """A template that does not render it gets the reason the spine's worker would: clipped
    to the runner's limit, head and tail, under the feedback heading."""
    from loop_ops import LONG_REASON

    from refactory_dsh.run import REASON_LIMIT, _clipped

    model = FakeModel()
    result = nn.execute(_reloop_to_agent("loop_ops:long_reason_once", "Write it."), tmp_path, model=model)
    assert result.exit_ok, result.error
    second = model.prompts[1][1]
    assert f"## Feedback for this visit\n\n{_clipped(LONG_REASON)}\n\nWrite it." in second
    assert len(_clipped(LONG_REASON)) <= REASON_LIMIT


def test_a_clipped_reason_is_said_only_when_its_tail_is_too():
    """The head alone does not say a clipped reason: a prompt that renders only the start of
    the log still gets the reason under the heading."""
    from loop_ops import LONG_REASON

    from refactory_dsh.agent_delivery import FEEDBACK_HEADING, with_feedback
    from refactory_dsh.run import _clipped

    prompt = LONG_REASON[:4000]
    assert with_feedback(prompt, _clipped(LONG_REASON)).startswith(f"{FEEDBACK_HEADING}\n\n")


def test_a_nodes_inputs_are_listed_in_name_order(tmp_path):
    """The lowered graph keeps reads as a set, sorted; the walker lists them so too."""
    for name in ("b.txt", "a.txt"):
        (tmp_path / name).write_text(name)
    (tmp_path / "c").mkdir()
    graph = nn.Graph()
    graph.add(nn.Node("ask", "agent", prompt_template="go", reads=("b.txt", "c/", "a.txt"), writes=("x.txt",)))
    model = FakeModel()
    assert nn.execute(graph, tmp_path, model=model).exit_ok
    inputs = json.loads(model.prompts[0][1].split("<!-- refactory worker execution -->\n", 1)[1]
                        .split("\n", 1)[1].split("\n<!-- end refactory worker execution -->", 1)[0])["inputs"]
    assert [path.rsplit("/", 1)[-1] for path in inputs] == ["a.txt", "b.txt", "c"]


def test_a_gate_visit_records_its_reads_in_name_order(tmp_path):
    """The ledger's gate visit lists the reads its review fingerprint is taken over, sorted."""
    from refactory_dsh.steering import Ledger

    ledger = Ledger(tmp_path)
    ledger.visit("gate", ["design.md", "delivery.md"])
    assert 'reads: ["delivery.md", "design.md"]' in ledger.path.read_text()


# ── the run's files ──────────────────────────────────────────────────────────


def test_the_walker_leaves_the_spines_facts_and_status(tmp_path):
    """`facts.json` holds the run's facts, and the status file the nodes completed by the
    spine's rules: a reloop un-completes what it goes back over, which completes again
    after the gate (`Spine.uncompleteBetween`)."""
    model = FakeModel()
    result = nn.execute(_reloop_to_agent("loop_ops:explained_passes", "Write it."), tmp_path, model=model)
    assert result.exit_ok, result.error
    assert json.loads((tmp_path / "facts.json").read_text()) == {"passes": 3}
    status = _status(tmp_path)
    assert (status["status"], status["nodes"], status["running_nodes"]) == ("done", [], [])
    assert status["completed"] == ["gate", "write", "done"]
    assert status["run_id"] == str(tmp_path.resolve()) and status["at"].endswith("Z")
    # Like the spine's, it names its runtime and the process that owns the run.
    assert (status["runtime"], status["owner_pid"]) == ("walker", os.getpid())


def test_a_gates_verdict_is_the_runs_record_not_a_fact(tmp_path):
    """No `<gate>_verdict` fact: the verdicts are kept where the spine keeps its own."""
    result = nn.execute(_reloop_to_agent("loop_ops:explained_passes", "Write it."), tmp_path, model=FakeModel())
    assert "gate_verdict" not in result.facts
    assert result.gate_verdicts == {"gate": "proceed"}
    assert json.loads((tmp_path / PROGRESS).read_text())["gate_verdicts"] == {"gate": "proceed"}


def test_a_forks_branches_complete_in_branch_order(tmp_path):
    """The spine completes a fork's branches boundary by boundary in target order, whatever
    finishes first."""
    graph = nn.Graph()
    graph.add(nn.Node("fork", "fork", targets=("slow", "fast")))
    graph.add(nn.Node("slow", "op", call="loop_ops:pause"))
    graph.add(nn.Node("fast", "op", call="loop_ops:noop"))
    graph.add(nn.Node("join", "join", sources=("slow", "fast")))
    graph.add(nn.Node("end", "op", call="loop_ops:noop"))
    graph.connect("fork", "join")
    graph.connect("slow", "join")
    graph.connect("fast", "join")
    graph.connect("join", "end")
    result = nn.execute(graph, tmp_path)
    assert result.exit_ok, result.error
    assert _status(tmp_path)["completed"] == ["fork", "slow", "fast", "join", "end"]


def _progress(run) -> dict:
    return json.loads((run / PROGRESS).read_text())


def test_a_gate_out_of_reloops_is_recorded_halted_with_nothing_gone_back_over(tmp_path):
    """Past its cap the spine records the gate `halt`, keeps the nodes before it completed
    and hands nobody feedback (`Spine.verdict`); the walker's record says the same."""
    graph = _reloop_to_agent("loop_ops:explained_passes", "Write it.")
    graph.nodes["gate"].max_iterations = 1
    result = nn.execute(graph, tmp_path, model=FakeModel())
    assert result.exit_ok is False and "max_iterations" in (result.error or "")
    progress = _progress(tmp_path)
    assert (progress["completed"], progress["gate_verdicts"], progress["feedback"]) == (
        ["gate", "write"], {"gate": "halt"}, {})


def _with_fork_in_the_loop() -> nn.Graph:
    """pre, a fork of p and q, their join, then a gate that sends the run back to pre."""
    graph = nn.Graph()
    graph.add(nn.Node("pre", "op", call="loop_ops:noop"))
    graph.add(nn.Node("f", "fork", targets=("p", "q")))
    graph.add(nn.Node("p", "op", call="loop_ops:noop"))
    graph.add(nn.Node("q", "op", call="loop_ops:noop"))
    graph.add(nn.Node("j", "join", sources=("p", "q")))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:three_passes", max_iterations=4, produces=("passes",)))
    graph.add(nn.Node("end", "op", call="loop_ops:noop"))
    for source, target in (("pre", "f"), ("f", "j"), ("p", "j"), ("q", "j"), ("j", "gate")):
        graph.connect(source, target)
    graph.connect("gate", "pre", condition="retry")
    graph.connect("gate", "end", condition="proceed")
    return graph


def test_a_reloop_goes_back_over_a_fork_and_its_branches(tmp_path):
    """The spine un-completes a fork's branches with it (`uncompleteBetween` follows a
    fork's targets), so they complete again in order after the gate."""
    assert nn.execute(_with_fork_in_the_loop(), tmp_path).exit_ok
    assert _status(tmp_path)["completed"] == ["gate", "pre", "f", "p", "q", "j", "end"]


def test_a_reloop_forgets_the_verdict_of_a_gate_it_goes_back_over(tmp_path):
    """A gate between the reloop's target and its gate is un-completed and its verdict
    dropped, as the spine drops it, and so is every node of the cycle past that gate's own
    edges (dm-226 review: a reloop reopens exactly its cycle on both runtimes)."""
    graph = nn.Graph()
    graph.add(nn.Node("pre", "op", call="loop_ops:fail_on_second_visit", writes=("visits.json",)))
    graph.add(nn.Node("inner", "gate", evaluator="loop_ops:always_proceed"))
    graph.add(nn.Node("mid", "op", call="loop_ops:noop"))
    graph.add(nn.Node("outer", "gate", evaluator="loop_ops:reloop_twice", max_iterations=4, produces=("passes",)))
    graph.add(nn.Node("end", "op", call="loop_ops:noop"))
    graph.connect("pre", "inner")
    graph.connect("inner", "mid", condition="proceed")
    graph.connect("mid", "outer")
    graph.connect("outer", "pre", condition="reloop")
    graph.connect("outer", "end", condition="proceed")
    result = nn.execute(graph, tmp_path)
    assert result.exit_ok is False and "second visit" in (result.error or "")
    progress = _progress(tmp_path)
    assert (progress["completed"], progress["gate_verdicts"]) == (["outer"], {"outer": "reloop"})


class FailsSecondCall(FakeModel):
    """Answers once, then fails: a branch that stops right after its gate sent it back."""

    def complete(self, prompt: str, *, system: str = "") -> str:
        if self.prompts:
            raise RuntimeError("the model is down")
        return super().complete(prompt, system=system)


def test_a_resumed_fork_branch_gets_the_feedback_its_last_reloop_left(tmp_path):
    """The run's pending feedback is a branch's too: a branch whose reloop target failed
    before it ran gets the reason when the fork is resumed, as the spine's checkpoint
    keeps it for the node."""
    graph = nn.Graph()
    graph.add(nn.Node("f", "fork", targets=("write",)))
    graph.add(nn.Node("write", "agent", prompt_template="Write it.", writes=("answer.txt",)))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:explained_passes", max_iterations=4, produces=("passes",)))
    graph.add(nn.Node("j", "join", sources=("gate",)))
    graph.connect("f", "j")
    graph.connect("write", "gate")
    graph.connect("gate", "write", condition="retry")
    graph.connect("gate", "j", condition="proceed")
    stopped = nn.execute(graph, tmp_path, model=FailsSecondCall())
    assert stopped.exit_ok is False and _progress(tmp_path)["feedback"] == {"write": "pass 1 of 3"}
    model = FakeModel()
    resumed = nn.execute(graph, tmp_path, model=model, start="f")
    assert resumed.exit_ok, resumed.error
    assert "## Feedback for this visit\n\npass 1 of 3\n\nWrite it." in model.prompts[0][1]


def _user_gate_graph() -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node("first", "op", call="loop_ops:noop"))
    graph.add(nn.Node("gate", "gate", gate_kind="user", notes="Go on?"))
    graph.add(nn.Node("last", "op", call="loop_ops:noop"))
    graph.connect("first", "gate")
    graph.connect("gate", "last", condition="proceed")
    return graph


def test_a_walk_stopped_at_a_user_gate_says_so_and_resumes_with_its_record(tmp_path):
    """Stopped at the gate, the status is the spine's `refused` naming the gate; resumed,
    the run goes on with the nodes it completed."""
    stopped = nn.execute(_user_gate_graph(), tmp_path)
    assert stopped.gate_stop == "gate"
    status = _status(tmp_path)
    assert (status["status"], status["nodes"], status["completed"]) == ("refused", ["gate"], ["first"])
    assert status["wait_reason"] == "waiting for user gate: gate (gate)"
    assert "waiting for a person" in status["error"]
    resumed = nn.execute(_user_gate_graph(), tmp_path, start="gate", verdicts={"gate": "proceed"})
    assert resumed.exit_ok, resumed.error
    assert _status(tmp_path)["completed"] == ["first", "gate", "last"]
    assert resumed.gate_verdicts == {"gate": "proceed"}


def test_a_failed_node_is_named_in_the_status(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("first", "op", call="loop_ops:noop"))
    graph.add(nn.Node("broken", "op", call="loop_ops:boom"))
    graph.connect("first", "broken")
    result = nn.execute(graph, tmp_path)
    assert result.exit_ok is False
    status = _status(tmp_path)
    assert (status["status"], status["nodes"], status["completed"]) == ("failed", ["broken"], ["first"])
    assert status["last_failure"]["node"] == "broken"


# ── readers that ask whether a walked run is alive ──────────────────────────


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _walked_run_left_running(run, runtime: str = "walker"):
    """A run whose process died mid-node: its status still says it is executing. A stale
    spine checkpoint from an earlier run in the directory changes nothing."""
    (run / ".refactory").mkdir(parents=True, exist_ok=True)
    (run / PROGRESS).write_text(json.dumps({"node": "emit", "error": None, "outcome": "running"}))
    (run / ".refactory/spine-checkpoint.json").write_text(json.dumps({"owner_pid": os.getpid()}))
    (run / ".refactory/spine-status.json").write_text(json.dumps(
        {"runtime": runtime, "owner_pid": _dead_pid(), "status": "executing", "nodes": ["emit"]}))


def test_a_walked_run_whose_process_died_has_ended_for_its_worktree(tmp_path):
    from refactory_dsh import worktree

    _walked_run_left_running(tmp_path)
    assert worktree.state(tmp_path, _dead_pid()) == "ended"
    assert worktree.state(tmp_path) == "ended", "the status file's owner answers when no claimant does"
    # A driven run's status is the spine's word, whoever claimed it.
    _walked_run_left_running(tmp_path, runtime="spine")
    assert worktree.state(tmp_path, _dead_pid()) == "active"


def test_a_walked_run_whose_process_died_no_longer_holds_its_artifact(tmp_path):
    from refactory_dsh import repo_change

    _walked_run_left_running(tmp_path)
    assert repo_change._still_open(tmp_path, _dead_pid()) is None
    assert repo_change._still_open(tmp_path) is None, "the status file's owner answers when no claimant does"
    # A live owner keeps the name while it walks.
    status = json.loads((tmp_path / ".refactory/spine-status.json").read_text())
    (tmp_path / ".refactory/spine-status.json").write_text(json.dumps({**status, "owner_pid": os.getpid()}))
    assert repo_change._still_open(tmp_path) == "walking, last at 'emit'"


def test_the_console_says_a_walked_run_whose_process_died_died_and_offers_no_graph_controls(tmp_path):
    from refactory_dsh import console

    _walked_run_left_running(tmp_path)
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs/default_pack-design.graph.json").write_text("{}")
    frame = console.status(tmp_path)
    assert (frame["state"], frame["kind"]) == ("died", "walker")


def test_a_directory_walked_without_lowering_can_be_prepared_for_a_new_run(tmp_path):
    """A bare `nn.execute` (an outer-loop candidate, a user's graph) leaves facts and a
    status file but no declarations; preparing a run there archives them as any earlier
    run's are, instead of refusing the directory."""
    from refactory_dsh import invocations, registry
    from refactory_dsh.paths import REPO_ROOT

    run = (tmp_path / "run").resolve()
    assert nn.execute(_user_gate_graph(), run, verdicts={"gate": "proceed"}).exit_ok
    assert (run / "facts.json").exists() and not (run / ".refactory/declarations.json").exists()
    state = registry.resolve(REPO_ROOT, "default_pack/create")[1]().state_dict()
    invocations.prepare(state, run, name="default_pack-create", intent="build it",
                        transform=lambda workflow, stage_dir=None: workflow)
    assert (run / ".refactory/declarations.json").exists()
    assert not (run / ".refactory/spine-status.json").exists() and not (run / PROGRESS).exists()
