"""Both-runtimes record equivalence (F6 correction #5).

The walker and the TS spine produce records from ``make_item_record`` with
matching fields for the same items.  This test extends the data-node
equivalence path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import nn, task as task_module
from refactory_dsh.items import make_item_record

TASK_REF = "equiv_counting"
NAMES = ("alpha", "beta", "gamma")


class EquivCounting(task_module.Task):
    """Three instances, one fails."""

    name = TASK_REF

    def instances(self) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id="alpha", metadata={"label": "A"}),
            task_module.TaskInstance(id="beta", metadata={"label": "B"}),
            task_module.TaskInstance(id="gamma", metadata={"label": "C"}),
        ]

    def verify(self, instance, workspace):
        if instance.id == "gamma":
            return task_module.VerifyResult(passed=False, score=0.0, details="gamma fails")
        return task_module.VerifyResult(passed=True, score=1.0, details=f"ok-{instance.id}")


task_module.register_task(TASK_REF, EquivCounting)


def _probe_graph() -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=TASK_REF, writes=("current_item.json",), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "finish", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "finish")
    return graph


class EquivProbe(nn.Mode):
    name = "equiv_probe"

    def build(self) -> nn.Graph:
        return _probe_graph()

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


# ── test_records_equivalence (walker) ──────────────────────────────────────

def test_records_equivalence(tmp_path):
    """Walker produces records with the expected shape for pass, fail items."""
    from refactory_dsh.evaluators import WalkerEvaluator

    mode = EquivProbe()
    state = mode.state_dict()
    evaluator = WalkerEvaluator(project=Path("."), max_steps=2000)
    workdir = tmp_path / "walk"
    result = evaluator.evaluate(mode, state, workdir)

    assert len(result.item_records) == 3, f"expected 3 records, got {len(result.item_records)}"

    by_id = {r["id"]: r for r in result.item_records}

    # All records have the same keys.
    expected_keys = {"id", "passed", "score", "exchanges", "prediction", "label",
                     "label_text", "fail_reason", "error_kind", "error_text", "metadata"}
    for record in result.item_records:
        assert set(record.keys()) == expected_keys, f"unexpected keys: {set(record.keys()) - expected_keys}"

    # Alpha passes.
    alpha = by_id["alpha"]
    assert alpha["passed"] is True
    assert alpha["score"] == 1.0
    assert alpha["label"] == "A"
    assert alpha["error_kind"] is None

    # Gamma fails verification.
    gamma = by_id["gamma"]
    assert gamma["passed"] is False
    assert gamma["score"] == 0.0
    assert gamma["label"] == "C"
    assert gamma["fail_reason"] == "gamma fails"


# ── test_spine_item_records_in_status ──────────────────────────────────────

def test_spine_item_records_in_status():
    """The TS spine's recordStatus includes item_records when present.

    This is a structural test: the spine code adds item_records to status JSON
    when records exist. The full integration test requires a DSH runtime.
    """
    # This test verifies the Python side produces records that the spine
    # would consume: the make_item_record function produces a JSON-serializable
    # dict that matches the TS ItemRecord interface.
    record = make_item_record(
        "test-item",
        passed=True,
        score=0.9,
        exchanges=[{"node": "agent1", "prompt": "question", "response": "answer"}],
        prediction="answer",
        label="correct",
        label_text="The correct answer",
        fail_reason=None,
        error_kind=None,
        error_text=None,
        metadata={"difficulty": "easy"},
    )
    # The record must be JSON-serializable.
    serialized = json.dumps(record)
    restored = json.loads(serialized)

    # The shape matches what the TS ItemRecord interface expects.
    assert restored["id"] == "test-item"
    assert restored["passed"] is True
    assert isinstance(restored["exchanges"], list)
    assert restored["exchanges"][0]["node"] == "agent1"
    assert restored["prediction"] == "answer"
    assert restored["label"] == "correct"
    assert restored["label_text"] == "The correct answer"
    assert restored["fail_reason"] is None
    assert restored["error_kind"] is None
    assert restored["metadata"]["difficulty"] == "easy"
