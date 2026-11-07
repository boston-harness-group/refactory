"""V17: grader-derived data stays on the grader side, whatever the Task or domain.

The provenance set is computed from the graph and the declarations -- what a data pass
publishes, the runtime's grading files, a Task's ``grader_outputs``, a mode's
``grader_nodes`` -- and propagated through every producer -> consumer edge (a prompt
slot, a bound field, ``feedback_keys``, reads, required_reads). No fact name is special.
Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.items import GRADER_FILES
from refactory_dsh.nn.validate import grader_derived, grader_issues, validate_state
from refactory_dsh.task import Task, TaskInstance, register_task


class _Graded(Task):
    name = "provenance_graded"
    grader_outputs = ("judge_notes.txt",)

    def instances(self, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        return [TaskInstance(id="a")]


register_task(_Graded.name, _Graded)


def _graph(*extra: nn.Node, task: str = _Graded.name, publishes: str = "verdicts") -> nn.Graph:
    """A data pass over one op, publishing its graded records as ``publishes`` (any
    name), then ``extra`` nodes in a line after it."""
    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="work", subgraph_exit="work",
                      task_ref=task, writes=("current_item.json",), produces=(publishes,)))
    graph.add(nn.Node("work", "op", call="x:y", reads=("current_item.json",),
                      writes=("answer.txt",), produces=("answer",)))
    previous = "data"
    for node in extra:
        graph.add(node)
        graph.connect(previous, node.id)
        previous = node.id
    return graph


def _agent(node_id: str, prompt: str = "Go.", **fields: Any) -> nn.Node:
    fields.setdefault("writes", (f"{node_id}.txt",))
    return nn.Node(node_id, "agent", prompt_template=prompt, system_prompt_template="You.",
                   tool_policy="none", role="solver", **fields)


def _op(node_id: str, **fields: Any) -> nn.Node:
    return nn.Node(node_id, "op", call="x:y", **fields)


def _flagged(graph: nn.Graph, knobs: dict[str, Any] | None = None, **scope: Any) -> set[str]:
    return {issue.node for issue in grader_issues(graph, knobs or {}, **scope)}


def test_whatever_a_data_pass_publishes_is_grader_derived_by_any_name():
    for name in ("verdicts", "item_scores", "results"):
        graph = _graph(_agent("peek", f"See {{{{{name}}}}}."), publishes=name)
        assert _flagged(graph) == {"peek"}, name
    # The pass's own per-item input is not the grade.
    assert ("file", "current_item.json") not in grader_derived(_graph(), {})


def test_taint_propagates_through_ops_files_and_facts():
    """A protected op may read the grade, but what it writes is the grade too: an agent
    reading its file, or a fact an op made from that file, is refused."""
    graph = _graph(_op("tally", reads=(), produces=("tally",), writes=("summary.txt",),
                       feedback_keys=("verdicts",)),
                   _op("relay", reads=("summary.txt",), produces=("relayed",)),
                   _agent("peek", "Use {{relayed}}."))
    assert _flagged(graph, grader_side=("tally",)) == {"relay", "peek"}
    derived = grader_derived(graph, {})
    assert {("file", "summary.txt"), ("fact", "tally"), ("fact", "relayed")} <= derived


def test_the_runtime_grading_files_and_a_tasks_grader_outputs():
    for name in GRADER_FILES:
        assert _flagged(_graph(_op("reader", reads=(name,)))) == {"reader"}, name
    graph = _graph(_agent("peek", reads=("judge_notes.txt",)))
    assert _flagged(graph) == {"peek"}, "a Task's grader_outputs"
    assert _flagged(_graph(_agent("peek", "{{judge_notes.txt}}"))) == set()


def test_a_declared_grader_nodes_outputs_and_the_grader_side_exception():
    graph = _graph(_agent("judge", writes=("judgement.txt",), produces=("judgement",)),
                   _agent("peek", "Use {{judgement}}."))
    assert _flagged(graph) == set(), "an ordinary agent's output is not the grade"
    assert _flagged(graph, grader_side=("judge",), graders=("judge",)) == {"peek"}
    reader = _graph(_agent("peek", "See {{verdicts}}."))
    assert _flagged(reader, grader_side=("peek",)) == set(), "a protected node may read it"


def test_a_slot_through_a_bound_field_is_consumption():
    node = _agent("peek", "Go.", knobs=("peek_prompt",),
                  field_bindings=(("prompt_template", "peek_prompt"),))
    assert _flagged(_graph(node), {"peek_prompt": "Look at {{verdicts}}"}) == {"peek"}
    assert _flagged(_graph(node), {"peek_prompt": "Look again"}) == set()


def test_validate_state_reports_it_and_every_shipped_mode_is_clean():
    graph = _graph(_agent("peek", "See {{verdicts}}."))
    state = {"graph": graph.to_state(), "knobs": {}}
    assert any(i.code == "V17" for i in validate_state(state, resolve_calls=False))
    from test_lowering_smoke import _mode_classes

    classes = [c[-1] if isinstance(c, tuple) else c for c in _mode_classes()]
    assert len(classes) == 22
    for cls in classes:
        mode = cls()
        issues = validate_state(mode.state_dict(), invariants=mode.invariants(),
                                **mode.grader_scope())
        assert not [i for i in issues if i.code == "V17"], (cls.name, issues)
