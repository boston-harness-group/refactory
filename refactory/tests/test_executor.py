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
    on both runtimes.
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
    assert json.loads((workdir / "seen.json").read_text())["ids"] == ["a", "b"], (
        "the subgraph did not run once per item"
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
    assert (workdir / "setup-a").exists() and (workdir / "setup-b").exists(), (
        "the Task's setup did not run per item"
    )
    assert result.facts["item_scores"] == [
        {"id": "a", "passed": True, "score": 1.0},
        {"id": "b", "passed": True, "score": 0.5},
    ]


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
