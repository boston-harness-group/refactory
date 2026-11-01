"""Per-item records (F6): every evaluated item gets a record an optimizer can read.

Tests that the extended ``make_item_record`` produces the required shape, that
records reach the optimizer via ``ProposalContext.item_records``, and that val/test
records never leak into optimizer context.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import nn, task as task_module
from refactory_dsh.items import make_item_record
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate

MODE = "holdout_probe/probe"
TASK_REF = "records_counting"
NAMES = ("alpha", "beta", "gamma", "delta")


class RecordsCounting(task_module.Task):
    """Four instances with label metadata, judged on whether the pass ran."""

    name = TASK_REF

    def instances(self) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id=name, metadata={"label": name.upper(), "label_text": f"Label for {name}"})
            for name in NAMES
        ]

    def verify(self, instance, workspace):
        if instance.id == "gamma":
            return task_module.VerifyResult(passed=False, score=0.0, details="gamma always fails")
        return task_module.VerifyResult(passed=True, score=1.0, details=f"predicted {instance.id}")


task_module.register_task(TASK_REF, RecordsCounting)


# ── test_item_record_shape ─────────────────────────────────────────────────

def test_item_record_shape():
    """The extended ``make_item_record`` has all required keys including F6 fields."""
    from refactory_dsh.task import VerifyResult
    record = make_item_record(
        "item-1",
        VerifyResult(passed=True, score=1.0, details="answer"),
        exchanges=[{"node": "agent", "prompt": "hello", "response": "world"}],
        prediction="answer",
        label="correct",
        label_text="Label text",
        fail_reason=None,
        metadata={"tag": "test"},
    )
    # All required keys are present.
    for key in ("id", "passed", "score", "exchanges", "prediction", "label",
                "label_text", "fail_reason", "error_kind", "error_text", "metadata"):
        assert key in record, f"missing key: {key}"
    assert record["id"] == "item-1"
    assert record["passed"] is True
    assert record["score"] == 1.0
    assert len(record["exchanges"]) == 1
    assert record["exchanges"][0]["node"] == "agent"
    assert record["prediction"] == "answer"
    assert record["label"] == "correct"
    assert record["metadata"] == {"tag": "test"}

    # Round-trips through JSON.
    restored = json.loads(json.dumps(record))
    assert restored == record


# ── test_walker_item_records ───────────────────────────────────────────────

def _probe_graph_for_records(task_ref: str) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=task_ref, writes=("current_item.json",), produces=("item_scores",),
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


class RecordsProbe(nn.Mode):
    """A scored mode bound to the RecordsCounting Task."""

    name = "records_probe"

    def build(self) -> nn.Graph:
        return _probe_graph_for_records(TASK_REF)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


def test_walker_item_records(tmp_path):
    """Walker evaluator produces per-item records with F6 fields."""
    from refactory_dsh.evaluators import WalkerEvaluator

    mode = RecordsProbe()
    state = mode.state_dict()
    evaluator = WalkerEvaluator(project=Path("."), max_steps=2000)
    workdir = tmp_path / "walk"
    result = evaluator.evaluate(mode, state, workdir)

    assert result.item_records, "no item records produced"
    assert (workdir / "item_records.json").exists(), "item_records.json not written"

    written = json.loads((workdir / "item_records.json").read_text())
    assert len(written) == len(result.item_records)

    ids = {r["id"] for r in result.item_records}
    assert ids == set(NAMES), f"expected all item ids, got {ids}"

    for record in result.item_records:
        assert "exchanges" in record
        assert "prediction" in record
        assert "label" in record
        assert "label_text" in record
        assert "fail_reason" in record
        assert "error_kind" in record
        assert "error_text" in record
        assert "metadata" in record
        assert isinstance(record["exchanges"], list)

    # Gamma fails verification; it should have a fail_reason.
    gamma = next(r for r in result.item_records if r["id"] == "gamma")
    assert gamma["passed"] is False
    assert gamma["fail_reason"] is not None
    assert "gamma always fails" in gamma["fail_reason"]
    assert gamma["label"] == "GAMMA"

    # Alpha passes; no fail_reason, has prediction.
    alpha = next(r for r in result.item_records if r["id"] == "alpha")
    assert alpha["passed"] is True
    assert alpha["fail_reason"] is None
    assert alpha["label"] == "ALPHA"


# ── test_records_reach_optimizer_train_only ─────────────────────────────────

def _patch_mode(mode_class):
    import contextlib
    import refactory_dsh.registry as registry

    class FakePackage:
        name = "fake"
        description = ""
        def mode(self, name):
            return mode_class

    @contextlib.contextmanager
    def patched():
        original = registry.resolve
        registry.resolve = lambda project, identifier: (FakePackage(), mode_class)
        try:
            yield
        finally:
            registry.resolve = original

    return patched()


def test_records_reach_optimizer_train_only(tmp_path):
    """Train records reach the optimizer via ProposalContext.item_records."""
    with _patch_mode(RecordsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        loop.forward()
        ctx = loop._context()

    assert ctx.item_records, "item_records should be non-empty for the parent candidate"
    for record in ctx.item_records:
        assert "exchanges" in record
        assert "prediction" in record
        assert "label" in record
        assert "fail_reason" in record


# ── test_val_test_records_never_reach_optimizer ────────────────────────────

def test_val_test_records_never_reach_optimizer(tmp_path):
    """Val and test records never appear in ProposalContext.item_records.

    The optimizer's ``item_records`` come from the loop's private train
    records storage.  Candidate no longer has a ``parts`` attribute, so
    ``candidate.parts["val"]`` is an AttributeError — the stronger invariant.
    """
    import json as _json

    with _patch_mode(RecordsProbe):
        # Write a manifest so val evaluation can find items.
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(_json.dumps({
            "train": list(NAMES),
            "val": list(NAMES),
        }))
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                parts_manifest=str(manifest_path),
            ),
            project=Path("."),
        )
        loop.init()
        loop.forward()

        # Evaluate the seed on val.
        seed = loop.population[0]
        loop._evaluate_part(seed, "val", None, None)

        ctx = loop._context()

    # Train records should be present.
    assert ctx.item_records, "item_records should contain train records"

    # Candidate has no `parts` attribute — the stronger invariant.
    assert not hasattr(seed, "parts"), (
        "Candidate should not have a 'parts' attribute"
    )

    # The context records must be exactly the train records from loop-private storage.
    train_records = list(loop._train_item_records.get(seed.id, []))
    assert ctx.item_records == train_records, (
        "context item_records must be exactly the train records, not val/test"
    )
