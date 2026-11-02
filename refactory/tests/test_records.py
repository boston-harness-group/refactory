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


# ── test_failed_item_record_has_full_shape ─────────────────────────────────

def test_failed_item_record_has_full_shape(tmp_path):
    """The spine's failed-item path produces a record with full shape.

    The failed_reason branch in command_end must produce exchanges, label,
    label_text, metadata, and fail_reason.  Reverting the parity fix makes
    the record bare (missing those fields).
    """
    from refactory_dsh.items import command_begin, command_end, record_exchange
    from refactory_dsh.nn.executor import CURRENT_ITEM, load_items

    # Build a graph with a data node bound to RecordsCounting.
    graph = _probe_graph_for_records(TASK_REF)
    data_node = None
    for node in graph.nodes.values():
        if node.kind == "data":
            data_node = node
            break
    assert data_node is not None

    workdir = tmp_path / "failed"
    workdir.mkdir(parents=True, exist_ok=True)
    task_module.write_subset(workdir, "train", 0.0)
    task_module.make_manifest(workdir, train_ids=list(NAMES))
    decl = {data_node.id: {
        "task_ref": data_node.task_ref,
        "produces": list(data_node.produces),
    }}
    (workdir / "declarations.json").write_text(json.dumps(decl))

    # Begin one item, write an exchange, then end with a failure.
    command_begin(data_node, workdir, 0)
    record_exchange(workdir, "agent_node", "hello", "world")
    record = command_end(data_node, workdir, 0,
                         failed_reason="subgraph failed",
                         failed_kind="RuntimeError")

    # The record must have the full shape.
    assert record is not None
    for key in ("exchanges", "label", "label_text", "metadata", "fail_reason"):
        assert key in record, f"missing key: {key}"
    assert record["fail_reason"] == "subgraph failed"
    assert record["exchanges"] == [{"node": "agent_node", "prompt": "hello", "response": "world"}]
    assert record["label"] == "ALPHA"  # First item is "alpha" with label "ALPHA"
    assert isinstance(record["metadata"], dict)


# ── test_prediction_on_pass_and_fail ───────────────────────────────────────

def test_prediction_on_pass_and_fail(tmp_path):
    """Prediction uses VerifyResult.prediction when set; falls back to
    verdict.details on pass and '' on fail when the Task gives none.

    Reverting the prediction fix makes passing items use verdict.details
    regardless and failing items always get None/empty.
    """
    from refactory_dsh.items import command_begin, command_end
    from refactory_dsh.nn.executor import load_items

    graph = _probe_graph_for_records(TASK_REF)
    data_node = None
    for node in graph.nodes.values():
        if node.kind == "data":
            data_node = node
            break
    assert data_node is not None

    workdir = tmp_path / "pred"
    workdir.mkdir(parents=True, exist_ok=True)
    task_module.write_subset(workdir, "train", 0.0)
    task_module.make_manifest(workdir, train_ids=list(NAMES))
    decl = {data_node.id: {
        "task_ref": data_node.task_ref,
        "produces": list(data_node.produces),
    }}
    (workdir / "declarations.json").write_text(json.dumps(decl))

    # Alpha passes: prediction should be verdict.details ("predicted alpha")
    # since the Task sets no VerifyResult.prediction.
    command_begin(data_node, workdir, 0)
    alpha_record = command_end(data_node, workdir, 0)
    assert alpha_record["prediction"] == "predicted alpha", (
        f"passing item without Task prediction should use verdict.details, got {alpha_record['prediction']!r}"
    )

    # Gamma fails: prediction should be "" when Task gives none.
    command_begin(data_node, workdir, 2)
    gamma_record = command_end(data_node, workdir, 2)
    assert gamma_record["prediction"] == "", (
        f"failing item without Task prediction should get '', got {gamma_record['prediction']!r}"
    )


# ── test_prediction_from_task_on_fail ──────────────────────────────────────

class PredictionOnFailTask(task_module.Task):
    """A Task that sets VerifyResult.prediction on failure."""

    name = "pred_fail_task"

    def instances(self):
        return [
            task_module.TaskInstance(id="alpha", metadata={"label": "A"}),
            task_module.TaskInstance(id="beta", metadata={"label": "B"}),
        ]

    def verify(self, instance, workspace):
        if instance.id == "beta":
            return task_module.VerifyResult(
                passed=False, score=0.0,
                details="beta fails",
                prediction="beta_prediction",
            )
        return task_module.VerifyResult(
            passed=True, score=1.0,
            details="ok-alpha",
            prediction="alpha_prediction",
        )


task_module.register_task("pred_fail_task", PredictionOnFailTask)


def test_prediction_from_task_on_fail(tmp_path):
    """A failing item whose Task sets verdict.prediction has that prediction
    in the record, not ''.

    Reverting the VerifyResult.prediction read makes failing items always
    get prediction=''.
    """
    from refactory_dsh.items import command_begin, command_end
    from refactory_dsh import nn

    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref="pred_fail_task", writes=("current_item.json",), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
    ))
    graph.connect("data", "data")  # not used, just need the nodes

    data_node = graph.nodes["data"]

    workdir = tmp_path / "pred_fail"
    workdir.mkdir(parents=True, exist_ok=True)
    task_module.write_subset(workdir, "train", 0.0)
    task_module.make_manifest(workdir, train_ids=["alpha", "beta"])
    decl = {data_node.id: {
        "task_ref": data_node.task_ref,
        "produces": list(data_node.produces),
    }}
    (workdir / "declarations.json").write_text(json.dumps(decl))

    # Alpha passes with Task-set prediction.
    command_begin(data_node, workdir, 0)
    alpha_record = command_end(data_node, workdir, 0)
    assert alpha_record["prediction"] == "alpha_prediction", (
        f"passing item should use Task prediction, got {alpha_record['prediction']!r}"
    )

    # Beta fails with Task-set prediction.
    command_begin(data_node, workdir, 1)
    beta_record = command_end(data_node, workdir, 1)
    assert beta_record["prediction"] == "beta_prediction", (
        f"failing item with Task prediction should use it, got {beta_record['prediction']!r}"
    )
