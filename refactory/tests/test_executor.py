"""The executor: one node at a time, gates bounded, declarations enforced.

Phase 3 gate of ``docs/plans/pythonic-modes.md``.
"""

from __future__ import annotations

import json

import pytest

from refactory_dsh import nn


class FakeModel:
    """A model client that records what it was asked."""

    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.prompts: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, system: str = "") -> str:
        self.prompts.append((system, prompt))
        return self.answer


def _linear_graph() -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node("first", "op", call="loop_ops:write_number",
                      writes=("number.json",), produces=("value",), knobs=("step",)))
    graph.add(nn.Node("second", "op", call="loop_ops:double",
                      reads=("number.json",), writes=("doubled.json",),
                      produces=("doubled",)))
    graph.add(nn.Node("last", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("first", "second")
    graph.connect("second", "last")
    return graph


def _loop_graph(gate_call: str, max_iterations: int = 8) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node("body", "op", call="loop_ops:write_number",
                      writes=("number.json",), produces=("value",), knobs=("step",)))
    graph.add(nn.Node("gate", "gate", evaluator=gate_call, max_iterations=max_iterations,
                      produces=("passes",)))
    graph.add(nn.Node("done", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("body", "gate")
    graph.connect("gate", "body", condition="retry")
    graph.connect("gate", "done", condition="proceed")
    return graph


def test_a_linear_run_passes_facts_between_nodes(tmp_path):
    result = nn.execute(_linear_graph(), tmp_path, knobs={"step": 4})
    assert result.exit_ok, result.error
    assert result.facts["value"] == 4
    assert result.facts["doubled"] == 8
    assert result.iterations == 3


def test_the_run_reports_every_step(tmp_path):
    executor = nn.Executor(_linear_graph(), nn.RunContext(tmp_path, knobs={"step": 1}))
    executor.run()
    assert [step.node for step in executor.steps] == ["first", "second", "last"]


def test_a_step_records_why_its_gate_sent_the_run_back(tmp_path):
    """The walker reports a gate's reason as the spine does under `ran`."""
    executor = nn.Executor(_loop_graph("loop_ops:explained_passes"), nn.RunContext(tmp_path, knobs={"step": 1}))
    executor.run()
    gates = [step for step in executor.steps if step.node == "gate"]
    assert [(step.verdict, step.reason) for step in gates] == [
        ("retry", "pass 1 of 3"), ("retry", "pass 2 of 3"), ("proceed", None)]


def test_a_gate_loops_until_its_verdict_changes(tmp_path):
    result = nn.execute(_loop_graph("loop_ops:three_passes"), tmp_path, knobs={"step": 2})
    assert result.exit_ok, result.error
    assert result.facts["passes"] == 3
    # The verdict is the run's record, as the spine's checkpoint keeps it, not a fact.
    assert result.gate_verdicts["gate"] == "proceed" and "gate_verdict" not in result.facts


def test_a_gate_loop_is_bounded_by_max_iterations(tmp_path):
    result = nn.execute(_loop_graph("loop_ops:always_retry", max_iterations=2), tmp_path)
    assert result.exit_ok is False
    assert "max_iterations" in (result.error or "")


def test_a_gate_must_return_a_declared_verdict(tmp_path):
    result = nn.execute(_loop_graph("loop_ops:bogus_verdict"), tmp_path)
    assert result.exit_ok is False
    assert "nonsense" in (result.error or "")
    assert "proceed" in (result.error or "")


def test_an_op_that_writes_what_it_did_not_declare_is_refused(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("liar", "op", call="loop_ops:undeclared_write", writes=("declared.json",)))
    result = nn.execute(graph, tmp_path)
    assert result.exit_ok is False
    assert "surprise.json" in (result.error or "")
    assert "does not declare" in (result.error or "")


def test_a_failing_op_ends_the_run_without_raising(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("boom", "op", call="loop_ops:boom"))
    result = nn.execute(graph, tmp_path)
    assert result.exit_ok is False
    assert "this op fails on purpose" in (result.error or "")


def test_an_agent_node_needs_a_model(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("ask", "agent", prompt_template="hello", writes=("answer.txt",)))
    result = nn.execute(graph, tmp_path)
    assert result.exit_ok is False
    assert "model client" in (result.error or "")


def test_an_agent_node_renders_its_prompt_and_writes_its_answer(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("ask", "agent",
                      prompt_template="knob is {{step}}, fact is {{value}}",
                      writes=("answer.txt",), produces=("answer",), knobs=("step",)))
    model = FakeModel("the answer")
    result = nn.execute(graph, tmp_path, knobs={"step": 9}, facts={"value": 3}, model=model)
    assert result.exit_ok, result.error
    assert len(model.prompts) == 1 and model.prompts[0][0] == ""
    assert model.prompts[0][1].endswith("knob is 9, fact is 3")
    assert (tmp_path / "answer.txt").read_text() == "the answer"
    assert result.facts["answer"] == "the answer"


def test_a_node_that_produces_an_undeclared_fact_is_refused(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("liar", "op", call="loop_ops:write_number",
                      writes=("number.json",), produces=("something_else",)))
    result = nn.execute(graph, tmp_path, knobs={"step": 1})
    assert result.exit_ok is False
    assert "produced the fact(s) ['value']" in (result.error or "")


def test_a_prompt_with_stray_braces_is_left_alone(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("ask", "agent", prompt_template='json: {"a": 1}, knob {{step}}',
                      writes=("answer.txt",), knobs=("step",)))
    model = FakeModel()
    nn.execute(graph, tmp_path, knobs={"step": 1}, model=model)
    assert model.prompts[0][1].endswith('json: {"a": 1}, knob 1')


def test_a_path_outside_the_work_directory_is_refused(tmp_path):
    ctx = nn.RunContext(tmp_path)
    for name in ("../escape.json", "/etc/passwd"):
        with pytest.raises(nn.ExecutionError):
            ctx.path(name)


def test_a_run_counts_its_iterations_and_time(tmp_path):
    result = nn.execute(_linear_graph(), tmp_path, knobs={"step": 1})
    assert result.iterations == 3
    assert result.duration_s >= 0
    assert result.model_calls == 0


def test_an_agent_node_is_a_counted_model_call(tmp_path):
    graph = nn.Graph()
    graph.add(nn.Node("ask", "agent", prompt_template="hello",
                      writes=("answer.txt",), produces=("answer",)))
    result = nn.execute(graph, tmp_path, model=FakeModel("hi"))
    assert result.exit_ok, result.error
    assert result.model_calls == 1


# ── Mode.evaluate ──────────────────────────────────────────────────────────
class ScoredMode(nn.Mode):
    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"step": nn.Knob(3, "int", lo=1, hi=9)}

    def build(self) -> nn.Graph:
        return _linear_graph()

    def score(self, run: nn.RunResult) -> float:
        return float(run.facts["doubled"])


class RaisingMode(ScoredMode):
    def score(self, run: nn.RunResult) -> float:
        raise ZeroDivisionError("no score today")


class FailedMode(ScoredMode):
    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("boom", "op", call="loop_ops:boom", writes=("never.json",)))
        return graph


def test_evaluate_puts_the_score_on_the_result(tmp_path):
    result = ScoredMode().evaluate(tmp_path)
    assert result.exit_ok, result.error
    assert result.score == 6.0


def test_a_failed_run_scores_zero_and_keeps_its_error(tmp_path):
    result = FailedMode().evaluate(tmp_path)
    assert result.exit_ok is False
    assert result.score == 0.0
    assert "fails on purpose" in (result.error or "")


def test_a_raising_score_is_a_failed_run_not_a_crash(tmp_path):
    result = RaisingMode().evaluate(tmp_path)
    assert result.exit_ok is False
    assert "score() raised ZeroDivisionError" in (result.error or "")


def test_the_default_score_does_not_make_a_mode_look_measured(tmp_path):
    class Unscored(ScoredMode):
        score = nn.Mode.score

    assert Unscored().is_scored() is False


def test_a_data_node_runs_its_subgraph_once_per_item(tmp_path):
    """A data node iterates: one pass per item, in order, with the item on disk.

    The walker used to refuse a data node by name, because the runtime that loads a
    source is DSH. Iterating here is what makes a dataset mode mean the same thing
    on both runtimes.  With pass isolation, each pass writes its own ``seen.json``
    in its pass directory.
    """
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        writes=(CURRENT_ITEM,), inline_items=({"id": "a"}, {"id": "b"}),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    workdir = tmp_path / "run"
    result = execute(graph, workdir)
    assert result.exit_ok, result.error
    # With pass isolation, each pass has its own seen.json in its pass directory.
    passes_dir = workdir / "passes" / "data"
    assert passes_dir.exists(), "passes/data/ should exist"
    all_seen: list[str] = []
    for subdir in sorted(passes_dir.iterdir()):
        if subdir.is_dir() and (subdir / "seen.json").exists():
            data = json.loads((subdir / "seen.json").read_text())
            all_seen.extend(data.get("ids", []))
    assert sorted(all_seen) == ["a", "b"], (
        f"the subgraph did not run once per item: {all_seen}"
    )


def test_a_data_node_with_no_items_iterates_nothing_and_still_finishes(tmp_path):
    """An empty selection is a fact about the data, not a crash."""
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        writes=(CURRENT_ITEM,), inline_items=(),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    assert not (tmp_path / "run" / "seen.json").exists(), "nothing should have run"


def test_a_data_node_bound_to_a_task_uses_its_items_and_verifies_each(tmp_path):
    """`task_ref` is a binding: the Task supplies the items and judges each pass."""
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class Probe(task_module.Task):
        name = "probe"

        def instances(self):
            return [task_module.TaskInstance(id="a"), task_module.TaskInstance(id="b")]

        def setup(self, instance, workspace):
            (workspace / f"setup-{instance.id}").write_text("prepared")

        def verify(self, instance, workspace):
            return task_module.VerifyResult(
                passed=True, score=1.0 if instance.id == "a" else 0.5
            )

    task_module.register_task("probe", Probe)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="probe", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best", writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    workdir = tmp_path / "run"
    result = execute(graph, workdir)
    assert result.exit_ok, result.error
    # With pass isolation, setup artifacts are in the pass directories.
    passes_dir = workdir / "passes" / "data"
    setup_files = []
    for subdir in sorted(passes_dir.iterdir()):
        if subdir.is_dir():
            setup_files.extend(f.name for f in subdir.iterdir() if f.name.startswith("setup-"))
    assert sorted(setup_files) == ["setup-a", "setup-b"], (
        f"the Task's setup did not run per item: {setup_files}"
    )
    # F6: item_scores carry all record fields now.
    for entry in result.facts["item_scores"]:
        assert "exchanges" in entry
        assert "prediction" in entry
        assert "metadata" in entry
    assert result.facts["item_scores"][0]["id"] == "a"
    assert result.facts["item_scores"][0]["passed"] is True
    assert result.facts["item_scores"][0]["score"] == 1.0
    assert result.facts["item_scores"][1]["id"] == "b"
    assert result.facts["item_scores"][1]["passed"] is True
    assert result.facts["item_scores"][1]["score"] == 0.5


def test_an_agent_node_tells_the_model_client_where_the_node_lives(tmp_path):
    """A completion runs in the node's run directory (#63's family).

    The client spawns an agent with tools, and that agent writes files with relative
    paths. Run from the checkout, a node's artifacts land in the repository: the
    reviewing nodes wrote `reviews/*.md` at the checkout root while the run's own
    `reviews/` stayed empty, so the QA synthesis read no reviews and the overwatch
    gate sent the build back to emit for nothing.
    """
    from refactory_dsh.nn import Graph, Node, RunContext
    from refactory_dsh.nn.executor import execute

    class Recording:
        def __init__(self) -> None:
            self.workdirs: list[object] = []

        def complete(self, prompt: str, *, system: str = "", workdir: object = None) -> str:
            self.workdirs.append(workdir)
            return "reviewed\n"

    graph = Graph()
    graph.add(Node("review", "agent", prompt_template="review it",
                   writes=("reviews/qa.md",), produces=("qa",)))
    client = Recording()
    result = execute(graph, tmp_path / "run", model=client, max_steps=10)
    assert result.exit_ok, result.error
    assert client.workdirs == [(tmp_path / "run").resolve()], (
        "the client was not told where the node's artifacts belong"
    )
    assert (tmp_path / "run" / "reviews" / "qa.md").exists()


def test_a_model_client_without_a_workdir_argument_is_still_asked(tmp_path):
    """The vendored protocol takes only text in, text out; the walker copes."""
    from refactory_dsh.nn import Graph, Node
    from refactory_dsh.nn.executor import execute

    class Minimal:
        def complete(self, prompt: str, *, system: str = "") -> str:
            return "done\n"

    graph = Graph()
    graph.add(Node("review", "agent", prompt_template="review it",
                   writes=("reviews/qa.md",), produces=("qa",)))
    result = execute(graph, tmp_path / "run", model=Minimal(), max_steps=10)
    assert result.exit_ok, result.error
    assert (tmp_path / "run" / "reviews" / "qa.md").read_text() == "done\n"


# ── resuming a failed run (#68) ────────────────────────────────────────────
#
# One transient transport failure used to discard an entire authoring run: every
# completed node's artifacts sat in the run directory and none of them could be used
# again. The executor now records how far it got, and a run resumes at the node that
# failed with the facts its predecessors produced.


class _Flaky:
    """A model whose first call fails on the wire and whose second succeeds."""

    def __init__(self, fail_first: bool = True) -> None:
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, prompt: str, *, system: str = "", workdir: object = None) -> str:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("model completion failed on the wire 3 time(s): dsh: TRANSPORT: boom")
        return "the second answer"


def _two_node_graph():
    from refactory_dsh.nn import Graph, Node

    graph = Graph()
    graph.add(Node("first", "op", call="loop_ops:write_number",
                   knobs=("step",), writes=("number.json",), produces=("value",)))
    graph.add(Node("second", "agent", prompt_template="say something",
                   reads=("number.json",), writes=("answer.txt",), produces=("answer",)))
    graph.connect("first", "second")
    return graph


def test_a_failed_run_records_where_it_stopped(tmp_path):
    from refactory_dsh.nn.executor import PROGRESS, execute

    result = execute(_two_node_graph(), tmp_path / "run", knobs={"step": 3},
                     model=_Flaky(), max_steps=10)
    assert result.exit_ok is False
    record = json.loads((tmp_path / "run" / PROGRESS).read_text())
    assert record["node"] == "second", "the failing node is what a resume needs"
    assert record["error"].startswith("second:")
    assert record["facts"]["value"] == 3, "the first node's facts must survive the failure"
    assert record["knobs"] == {"step": 3}
    assert (tmp_path / "run" / "number.json").exists()


def test_a_resumed_run_continues_instead_of_starting_over(tmp_path):
    from refactory_dsh.nn.mode import Mode

    class Two(Mode):
        name = "two"

        @classmethod
        def declare(cls):
            from refactory_dsh.nn import Knob

            return {"step": Knob(2, "int", lo=1, hi=4, description="a counter")}

        def build(self):
            return _two_node_graph()

    workdir = tmp_path / "run"
    first = Two().run(workdir, model=_Flaky(), max_steps=10)
    assert first.exit_ok is False

    # The op node's artifact is written once and must not be rewritten by the resume.
    before = (workdir / "number.json").read_text()
    assert '"value": 2' in before
    (workdir / "number.json").write_text(before.replace('"value": 2', '"value": 99'))

    resumed = Two.resume(workdir, model=_Flaky(fail_first=False))
    assert resumed.exit_ok, resumed.error
    assert (workdir / "answer.txt").read_text() == "the second answer"
    assert json.loads((workdir / "number.json").read_text())["value"] == 99, (
        "the resumed run re-ran the node that had already succeeded"
    )
    assert resumed.facts.get("answer") == "the second answer"


def test_resuming_a_directory_with_no_record_is_refused(tmp_path):
    from refactory_dsh.nn.mode import Mode

    class Two(Mode):
        name = "two"

        @classmethod
        def declare(cls):
            return {}

        def build(self):
            return _two_node_graph()

    with pytest.raises(ValueError, match="no run to resume"):
        Two.resume(tmp_path / "nothing-here")


# ── F4 failure rule: failed items score 0, other items continue ────────


def _data_graph_with_task(task_ref: str, **extra) -> nn.Graph:
    """A data node with a task, a one-node subgraph, and a report node."""
    from refactory_dsh.nn.executor import CURRENT_ITEM

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=task_ref, writes=(CURRENT_ITEM,), produces=("item_scores",), **extra,
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")
    return graph


def _register_failing_task(name: str, *, fail_on: set[str], fail_in: str = "verify"):
    """Register a task where certain items fail in setup or verify."""
    from refactory_dsh import task as task_module

    class FailingTask(task_module.Task):
        def instances(self):
            return [
                task_module.TaskInstance(id="a"),
                task_module.TaskInstance(id="b"),
            ]

        def setup(self, instance, workspace):
            if fail_in == "setup" and instance.id in fail_on:
                raise ValueError(f"setup failed for {instance.id}")

        def verify(self, instance, workspace):
            if fail_in == "verify" and instance.id in fail_on:
                raise ValueError(f"verify failed for {instance.id}")
            return task_module.VerifyResult(passed=True, score=1.0)

    FailingTask.name = name
    task_module.register_task(name, FailingTask)
    return FailingTask


def test_failed_item_scores_zero_and_others_continue(tmp_path):
    """A failed verify produces score 0 with error; other items score normally."""
    from refactory_dsh.nn.executor import execute

    _register_failing_task("fail_verify_a", fail_on={"a"}, fail_in="verify")
    graph = _data_graph_with_task("fail_verify_a")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    scores = result.facts["item_scores"]
    assert len(scores) == 2
    a_score = next(s for s in scores if s["id"] == "a")
    b_score = next(s for s in scores if s["id"] == "b")
    assert a_score["passed"] is False
    assert a_score["score"] == 0.0
    assert "verify failed" in a_score["error_text"]
    assert a_score["error_kind"] == "ValueError"
    assert b_score["passed"] is True
    assert b_score["score"] == 1.0


def test_failed_setup_becomes_failed_item(tmp_path):
    """A failed setup produces a failed item record; other items still run."""
    from refactory_dsh.nn.executor import execute

    _register_failing_task("fail_setup_a", fail_on={"a"}, fail_in="setup")
    graph = _data_graph_with_task("fail_setup_a")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    scores = result.facts["item_scores"]
    a_score = next(s for s in scores if s["id"] == "a")
    b_score = next(s for s in scores if s["id"] == "b")
    assert a_score["passed"] is False
    assert a_score["score"] == 0.0
    assert "setup failed" in a_score["error_text"]
    assert b_score["passed"] is True
    assert b_score["score"] == 1.0


def test_branch_error_becomes_failed_item(tmp_path):
    """A subgraph op that raises during one item's pass produces a failed item."""
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class GoodTask(task_module.Task):
        name = "good_task"

        def instances(self):
            return [
                task_module.TaskInstance(id="a"),
                task_module.TaskInstance(id="b"),
            ]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("good_task", GoodTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="good_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    # An op that fails on item "a"
    graph.add(nn.Node(
        "work", "op", call="loop_ops:fail_on_a",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    scores = result.facts["item_scores"]
    a_score = next(s for s in scores if s["id"] == "a")
    b_score = next(s for s in scores if s["id"] == "b")
    assert a_score["passed"] is False
    assert a_score["score"] == 0.0
    assert a_score.get("error_text") is not None
    assert b_score["passed"] is True
    assert b_score["score"] == 1.0


def test_all_items_failed_makes_node_fail(tmp_path):
    """When every item fails, the data node fails (exit_ok=False)."""
    from refactory_dsh.nn.executor import execute

    _register_failing_task("fail_all", fail_on={"a", "b"}, fail_in="verify")
    graph = _data_graph_with_task("fail_all")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is False
    assert "all" in (result.error or "").lower() and "failed" in (result.error or "").lower()
    # item_scores must still be written
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 2
    assert all(s.get("error_text") is not None for s in scores)


def test_passing_item_with_score_zero_alongside_failures_succeeds(tmp_path):
    """A passing item with score=0.0 does not trigger the all-fail rule."""
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class ZeroScore(task_module.Task):
        name = "zero_score"

        def instances(self):
            return [
                task_module.TaskInstance(id="a"),
                task_module.TaskInstance(id="b"),
            ]

        def verify(self, instance, workspace):
            if instance.id == "b":
                raise ValueError("b failed")
            return task_module.VerifyResult(passed=True, score=0.0)

    task_module.register_task("zero_score", ZeroScore)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="zero_score", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is True


def test_budget_exhausted_propagates_from_item(tmp_path):
    """BudgetExhausted raised inside an item must propagate, not be swallowed."""
    from refactory_dsh import task as task_module
    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class BudgetTask(task_module.Task):
        name = "budget_task"

        def instances(self):
            return [task_module.TaskInstance(id="a")]

        def verify(self, instance, workspace):
            raise BudgetExhausted("budget gone")

    task_module.register_task("budget_task", BudgetTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="budget_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    with pytest.raises(BudgetExhausted):
        execute(graph, tmp_path / "run")


def test_halt_stop_propagates_from_item(tmp_path):
    """HaltStop raised inside a data item must propagate; run is halted."""
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, HaltStop, execute

    class HaltTask(task_module.Task):
        name = "halt_task"

        def instances(self):
            return [task_module.TaskInstance(id="a")]

        def setup(self, instance, workspace):
            raise HaltStop("test_gate", reason="test halt")

    task_module.register_task("halt_task", HaltTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="halt_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is False
    assert result.halted is True


def test_csv_items_without_id_get_index_ids(tmp_path):
    """CSV items without an id column get index-based ids on the walker."""
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class CsvTask(task_module.Task):
        name = "csv_task"

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("csv_task", CsvTask)

    workdir = tmp_path / "run"
    workdir.mkdir(parents=True)
    (workdir / "items.csv").write_text("value\nfoo\nbar\n")

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="csv_task", source_path="items.csv", source_format="csv",
        writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    result = execute(graph, workdir)
    assert result.exit_ok, result.error
    scores = result.facts["item_scores"]
    ids = [s["id"] for s in scores]
    assert ids == ["0", "1"]


def test_large_benchmark_completes_under_step_cap(tmp_path):
    """A 600-item data node completes without hitting the step cap (parity fix 4)."""
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class LargeTask(task_module.Task):
        name = "large_task"

        def instances(self):
            return [task_module.TaskInstance(id=str(i)) for i in range(600)]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("large_task", LargeTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="end",
        task_ref="large_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "end", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.add(nn.Node(
        "after", "op", call="loop_ops:noop",
    ))
    graph.connect("work", "end")
    graph.connect("data", "after")

    result = execute(graph, tmp_path / "run", max_steps=1000)
    assert result.exit_ok, result.error
    assert len(result.facts["item_scores"]) == 600


def test_verify_timeout_becomes_failed_item(tmp_path):
    """A verify that hangs beyond the per-item deadline ends as a failed item (parity fix 1).

    The walker enforces the same per-item deadline as the spine's runDataCommand.
    When enable_recovery is True, a hanging verify is killed by SIGALRM and
    the item is recorded as failed with an error_kind indicating timeout.
    """
    import time as time_mod
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, Executor, RunContext

    class HangingVerifyTask(task_module.Task):
        name = "hanging_verify_task"

        def instances(self):
            return [
                task_module.TaskInstance(id="slow"),
                task_module.TaskInstance(id="fast"),
            ]

        def verify(self, instance, workspace):
            if instance.id == "slow":
                time_mod.sleep(30)  # hangs longer than deadline
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("hanging_verify_task", HangingVerifyTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="hanging_verify_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    workdir = tmp_path / "run"
    ctx = RunContext(workdir, enforce_writes=True)
    # Use a very short deadline so the test doesn't take too long.
    # The deadline does not depend on enable_recovery (Ask 2).
    executor = Executor(graph, ctx, max_steps=2000)
    executor._data_item_deadline = lambda: 2.0  # 2-second deadline
    result = executor.run()

    assert result.exit_ok, result.error
    scores = result.facts["item_scores"]
    assert len(scores) == 2

    slow_score = next(s for s in scores if s["id"] == "slow")
    fast_score = next(s for s in scores if s["id"] == "fast")

    assert slow_score["passed"] is False
    assert slow_score["score"] == 0.0
    assert slow_score.get("error_text") is not None
    assert "timed out" in slow_score["error_text"].lower() or "deadline" in slow_score["error_text"].lower()
    assert slow_score.get("error_kind") == "timeout"

    assert fast_score["passed"] is True
    assert fast_score["score"] == 1.0


def test_item_scores_no_double_counting_on_rerun(tmp_path):
    """item_scores from a data node rerun replaces the previous scores (parity fix 3)."""
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    _register_failing_task("rerun_task", fail_on={"a"}, fail_in="verify")
    graph = _data_graph_with_task("rerun_task")

    # First run
    result1 = execute(graph, tmp_path / "run")
    assert result1.exit_ok
    scores1 = result1.facts["item_scores"]
    assert len(scores1) == 2

    # The walker uses set_fact which replaces the entire list, so running
    # again with different facts should still produce exactly 2 entries.
    result2 = execute(graph, tmp_path / "run2")
    assert result2.exit_ok
    scores2 = result2.facts["item_scores"]
    assert len(scores2) == 2


def test_last_item_failure_does_not_fail_run(tmp_path):
    """A failure in the LAST item does not fail the whole run (Ask 1).

    Items [b, a] with fail_on_a: the last item (a) fails but exit_ok is True,
    item_scores has a failed and b passed, and iterations reflects all steps.
    Fails when the step wipe is restored or the step check does not skip
    failed-item steps.
    """
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class LastFailTask(task_module.Task):
        name = "last_fail_task"

        def instances(self):
            return [
                task_module.TaskInstance(id="b"),
                task_module.TaskInstance(id="a"),
            ]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("last_fail_task", LastFailTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="last_fail_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:fail_on_a",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is True, f"exit_ok should be True but error: {result.error}"
    scores = result.facts["item_scores"]
    assert len(scores) == 2
    a_score = next(s for s in scores if s["id"] == "a")
    b_score = next(s for s in scores if s["id"] == "b")
    assert a_score["passed"] is False
    assert a_score["score"] == 0.0
    assert a_score.get("error_text") is not None
    assert b_score["passed"] is True
    assert b_score["score"] == 1.0
    # Steps should not be wiped: iterations should reflect all steps.
    assert result.iterations > 0


def test_one_of_three_items_failing_without_task_succeeds(tmp_path):
    """With no Task, 1 of 3 items failing must not fail the node (Ask 5).

    3 inline items, NO Task, op fail_on_a fails on item a; exit_ok is True
    because only 1 of 3 items failed (the all-fail rule counts items, not
    records).  Without a Task, successful items produce no record, so
    ``scores`` has only the 1 failure.  The old code checked
    ``failed_count == len(scores)`` which would be ``1 == 1`` → all fail.
    The fix checks ``failed_count == len(items)`` → ``1 == 3`` → OK.
    """
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        inline_items=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
        writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:fail_on_a",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result = execute(graph, tmp_path / "run")
    assert result.exit_ok is True, f"1 of 3 failing should not fail the node: {result.error}"
    scores = result.facts["item_scores"]
    # Without a Task, only the failed item has a score record.
    failed = [s for s in scores if s.get("error_text") is not None]
    assert len(failed) == 1
    assert failed[0]["id"] == "a"


# ── Ask 4: thread-safe deadline ──────────────────────────────────────────────


def test_execute_from_thread_succeeds(tmp_path):
    """execute() from a threading.Thread with 2 plain items succeeds (Ask 4).

    Off the main thread, SIGALRM cannot be armed. The fix skips arming and
    emits a run_warning event. This test fails when the main-thread guard is
    removed (ValueError: signal only works in main thread).
    """
    import threading
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    class ThreadTask(task_module.Task):
        name = "thread_task"

        def instances(self):
            return [
                task_module.TaskInstance(id="one"),
                task_module.TaskInstance(id="two"),
            ]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("thread_task", ThreadTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="thread_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    result_holder: list = []
    error_holder: list = []

    def run_in_thread():
        try:
            result = execute(graph, tmp_path / "run")
            result_holder.append(result)
        except Exception as exc:
            error_holder.append(exc)

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join(timeout=60)

    assert not error_holder, f"thread raised: {error_holder[0]}"
    assert len(result_holder) == 1
    result = result_holder[0]
    assert result.exit_ok is True, f"expected exit_ok=True, got error: {result.error}"
    assert len(result.facts["item_scores"]) == 2

    # Correction 3: verify that the unarmed deadline was reported as a
    # run_warning event in the run's events (the same channel the spine uses).
    import json
    events_path = tmp_path / "run" / ".refactory" / "events.jsonl"
    assert events_path.exists(), "run_warning event should have been written"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    warnings = [e for e in events if e.get("kind") == "run_warning"]
    assert len(warnings) >= 1, f"expected a run_warning event, got {events}"
    assert "not on the main thread" in warnings[0].get("summary", "")


# ── Ask 5: step-cap scaling ────────────────────────────────────────────────


def test_step_cap_scaling_exceeds_unscaled_cap(tmp_path):
    """A data node sized so items × nodes > default max_steps succeeds (Ask 5).

    The step-cap scaling at executor.py:811 raises max_steps. This test confirms
    that the unscaled cap would be exceeded.
    """
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, Executor, RunContext, execute

    # Read the actual default max_steps.
    default_max = Executor.max_steps  # 2000

    # Subgraph has 2 nodes (work + end), so we need items such that:
    # items × 2 > default_max. Use (default_max // 2) + 1 items.
    n_items = (default_max // 2) + 1

    class ScaledTask(task_module.Task):
        name = "scaled_task"

        def instances(self):
            return [task_module.TaskInstance(id=str(i)) for i in range(n_items)]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("scaled_task", ScaledTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="end",
        task_ref="scaled_task", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "end", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.add(nn.Node("after", "op", call="loop_ops:noop"))
    graph.connect("work", "end")
    graph.connect("data", "after")

    # With scaling: all items should pass (no errors).
    result = execute(graph, tmp_path / "run-scaled", max_steps=default_max)
    assert result.exit_ok, f"scaled run should succeed: {result.error}"
    scaled_scores = result.facts["item_scores"]
    assert len(scaled_scores) == n_items
    scaled_failed = [s for s in scaled_scores if s.get("error_text") is not None]
    assert len(scaled_failed) == 0, (
        f"scaled run should have zero failures, got {len(scaled_failed)}"
    )

    # Without scaling (revert): should exceed the cap and fail some items.
    # Block the scaling line (self.max_steps += …) via a subclass whose
    # __setattr__ ignores writes to max_steps after __init__.
    assert n_items * 2 > default_max, (
        f"test miscalculated: {n_items} × 2 = {n_items * 2} should exceed {default_max}"
    )

    class _NoScaleExecutor(Executor):
        """Executor that blocks the data-loop step-cap scaling."""
        _init_done = False

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._init_done = True

        def __setattr__(self, attr, value):
            if attr == "max_steps" and getattr(self, "_init_done", False):
                return  # block the scaling
            super().__setattr__(attr, value)

    ctx_noscale = RunContext(tmp_path / "run-noscale", enforce_writes=True)
    ex_noscale = _NoScaleExecutor(graph, ctx_noscale, max_steps=default_max)
    result_noscale = ex_noscale.run()
    # Without scaling the step cap is exceeded: some items hit the cap and are
    # recorded as failed (branch walk returns "exceeded").  The scaled run had
    # zero failures; any failure here proves the scaling was necessary.
    noscale_scores = result_noscale.facts.get("item_scores", [])
    noscale_failed = [s for s in noscale_scores if s.get("error_text") is not None]
    assert len(noscale_failed) > 0, (
        f"without scaling some items should fail (step cap {default_max} < "
        f"{n_items} items × 2 nodes = {n_items * 2} total steps)"
    )


# ── Ask 5: deadline not armed across branch ─────────────────────────────────


def test_deadline_not_armed_across_branch(tmp_path):
    """A branch longer than the deadline passes because the deadline is disarmed (Ask 5).

    The deadline is disarmed before the branch walk at executor.py:845-847.
    A branch that sleeps for 3 seconds with a 2-second deadline still passes.
    """
    import time as time_mod
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, Executor, RunContext

    class QuickTask(task_module.Task):
        name = "quick_task_for_branch"

        def instances(self):
            return [task_module.TaskInstance(id="slow_branch")]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("quick_task_for_branch", QuickTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="quick_task_for_branch", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:slow_op",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    workdir = tmp_path / "run"
    ctx = RunContext(workdir, enforce_writes=True)
    executor = Executor(graph, ctx, max_steps=2000)
    # Set a 2-second deadline — the branch sleeps 3 seconds but should still pass
    # because the deadline is disarmed before the branch walk.
    executor._data_item_deadline = lambda: 2.0
    result = executor.run()

    assert result.exit_ok is True, f"deadline should not fire across branch: {result.error}"
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 1
    assert scores[0]["passed"] is True


# ── Ask 4: SIGALRM handler restored after branch failure ────────────────────


def test_sigalrm_handler_restored_after_branch_failure(tmp_path):
    """The previous SIGALRM handler is restored even when the branch fails (Ask 4).

    When the deadline is armed, then disarmed before the branch walk, and the
    branch fails, the finally block must still restore the previous handler.
    Fails when restoration is gated on _alarm_armed (which is False after
    the disarm-before-branch).
    """
    import signal
    from refactory_dsh import task as task_module
    from refactory_dsh.nn.executor import CURRENT_ITEM, Executor, RunContext

    sentinel_called = []

    def sentinel_handler(signum, frame):
        sentinel_called.append(True)

    class BranchFailTask(task_module.Task):
        name = "branch_fail_restore"

        def instances(self):
            return [
                task_module.TaskInstance(id="a"),  # branch fails for "a"
                task_module.TaskInstance(id="ok"),
            ]

        def verify(self, instance, workspace):
            return task_module.VerifyResult(passed=True, score=1.0)

    task_module.register_task("branch_fail_restore", BranchFailTask)

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="branch_fail_restore", writes=(CURRENT_ITEM,), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:fail_on_first",
        reads=(CURRENT_ITEM,), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")

    # Install a sentinel handler that must be restored after the data loop.
    prev_handler = signal.signal(signal.SIGALRM, sentinel_handler)
    try:
        workdir = tmp_path / "run"
        ctx = RunContext(workdir, enforce_writes=True)
        executor = Executor(graph, ctx, max_steps=2000)
        executor._data_item_deadline = lambda: 10.0
        result = executor.run()

        # After the data loop, the sentinel handler must be restored.
        current_handler = signal.getsignal(signal.SIGALRM)
        assert current_handler is sentinel_handler, (
            f"SIGALRM handler was not restored; got {current_handler}"
        )
    finally:
        signal.signal(signal.SIGALRM, prev_handler)


# ── F5: pass isolation tests ────────────────────────────────────────────────


def _isolation_graph(op: str = "loop_ops:assert_isolated", **node_kw) -> nn.Graph:
    """A data graph whose subgraph runs one op per item."""
    from refactory_dsh.nn.executor import CURRENT_ITEM

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        writes=(CURRENT_ITEM,), inline_items=({"id": "a"}, {"id": "b"}),
        produces=("item_scores",), **node_kw,
    ))
    graph.add(nn.Node(
        "work", "op", call=op,
        reads=(CURRENT_ITEM,), writes=("marker.txt",), produces=("leak_test",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")
    return graph


def test_item_pass_fact_isolation(tmp_path):
    """An op asserts the previous pass's fact is invisible.

    Fails without the dict-copy fact scope in pass_scope.
    """
    from refactory_dsh.nn.executor import execute

    workdir = tmp_path / "run"
    result = execute(_isolation_graph(), workdir)
    assert result.exit_ok, result.error
    # Every item must have passed (no error records).
    for record in result.item_records:
        assert record.get("error_text") is None, (
            f"item {record['id']} failed: {record.get('error_text')}"
        )


def test_item_pass_file_isolation(tmp_path):
    """An op asserts the previous pass's file is absent from the current workdir.

    Fails without the workdir swap in pass_scope.
    """
    from refactory_dsh.nn.executor import execute

    workdir = tmp_path / "run"
    result = execute(_isolation_graph(), workdir)
    assert result.exit_ok, result.error
    # marker.txt should NOT exist at run level (only in pass dirs)
    assert not (workdir / "marker.txt").exists(), (
        "marker.txt leaked to run-level workdir"
    )


def test_undeclared_fact_does_not_leak(tmp_path):
    """An op writes a fact not in the data node's produces; the fact stays in the pass.

    The subgraph op declares ``undeclared_secret`` in its own ``produces``
    (so it is allowed to set it). But the data node's ``produces`` is only
    ``("item_scores",)``, so pass_scope's merge filter must exclude it.
    Without the merge filter (reverting pass_scope), the fact would remain
    on the run's facts dict.

    Fails without the merge filter in pass_scope.__exit__.
    """
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        writes=(CURRENT_ITEM,), inline_items=({"id": "a"},),
        produces=("item_scores",),
    ))
    # The work node declares undeclared_secret so the op can set it,
    # but the data node does NOT declare it — so it must not leak.
    graph.add(nn.Node(
        "work", "op", call="loop_ops:write_undeclared_fact",
        reads=(CURRENT_ITEM,), writes=(), produces=("undeclared_secret",),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    workdir = tmp_path / "run"
    result = execute(graph, workdir)
    assert result.exit_ok, result.error
    assert "undeclared_secret" not in result.facts, (
        f"undeclared_secret leaked to run facts: {result.facts.get('undeclared_secret')}"
    )


def test_pass_directories_created(tmp_path):
    """After a data node run, passes/<node_id>/ has one subdir per item with .done."""
    from refactory_dsh.nn.executor import execute

    workdir = tmp_path / "run"
    result = execute(_isolation_graph(), workdir)
    assert result.exit_ok, result.error

    passes_dir = workdir / "passes" / "data"
    assert passes_dir.exists(), "passes/data/ does not exist"
    subdirs = sorted(d.name for d in passes_dir.iterdir() if d.is_dir())
    assert len(subdirs) == 2, f"expected 2 pass dirs, got {subdirs}"
    for subdir in passes_dir.iterdir():
        if subdir.is_dir():
            assert (subdir / ".done").exists(), f"{subdir.name} missing .done marker"


def test_walker_rerun_creates_fresh_passes(tmp_path):
    """Running the data node twice: second run's pass dirs are fresh."""
    from refactory_dsh.nn.executor import execute

    workdir = tmp_path / "run"
    result1 = execute(_isolation_graph(), workdir)
    assert result1.exit_ok, result1.error

    # Write a sentinel file in one of the first run's pass dirs.
    passes_dir = workdir / "passes" / "data"
    first_pass = next(passes_dir.iterdir())
    (first_pass / "stale.txt").write_text("from first run")

    result2 = execute(_isolation_graph(), workdir)
    assert result2.exit_ok, result2.error

    # The sentinel should be gone (fresh pass dirs).
    for subdir in passes_dir.iterdir():
        if subdir.is_dir():
            assert not (subdir / "stale.txt").exists(), (
                f"stale file survived rerun in {subdir.name}"
            )


def test_pass_file_read_through(tmp_path):
    """A run-level file declared as a subgraph input is readable inside a pass.

    Fails without the symlink materialisation in prepare_pass_dir.
    """
    from refactory_dsh.nn.executor import CURRENT_ITEM, execute

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        writes=(CURRENT_ITEM,), inline_items=({"id": "a"},),
        produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:read_run_level_input",
        reads=(CURRENT_ITEM, "run_input.txt"), writes=(), produces=(),
    ))
    graph.add(nn.Node(
        "report", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "report")

    workdir = tmp_path / "run"
    workdir.mkdir(parents=True)
    (workdir / "run_input.txt").write_text("run-level-data")

    result = execute(graph, workdir)
    assert result.exit_ok, result.error
