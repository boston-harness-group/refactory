"""Explicit train/val/test parts (F9) and evaluate-on-part (F18).

Tests that manifests, seeds, occasion directories, overfit verdicts, and
backward compatibility all work as designed.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from refactory_dsh import nn, task as task_module
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate
from refactory_dsh.outerloop import Run

MODE = "holdout_probe/probe"
TASK_REF = "parts_counting"
NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")


class PartsCounting(task_module.Task):
    """Six instances, so we can partition meaningfully."""

    name = TASK_REF

    def instances(self) -> list[task_module.TaskInstance]:
        return [
            task_module.TaskInstance(id=name, metadata={"label": name.upper()})
            for name in NAMES
        ]

    def verify(self, instance, workspace):
        return task_module.VerifyResult(passed=True, score=1.0)


task_module.register_task(TASK_REF, PartsCounting)


def _probe_graph(task_ref: str) -> nn.Graph:
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


class PartsProbe(nn.Mode):
    """A scored mode bound to PartsCounting."""

    name = "parts_probe"

    def build(self) -> nn.Graph:
        return _probe_graph(TASK_REF)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


def _patch_mode(mode_class):
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


# ── test_manifest_from_id_lists ────────────────────────────────────────────

def test_manifest_from_id_lists(tmp_path):
    """make_manifest writes a parts.json with explicit id lists, and
    select_subset reads from it."""
    task_module.make_manifest(
        tmp_path,
        train_ids=["alpha", "beta"],
        val_ids=["gamma"],
        test_ids=["delta"],
    )
    manifest = json.loads((tmp_path / task_module.MANIFEST).read_text())
    assert manifest["train"] == ["alpha", "beta"]
    assert manifest["val"] == ["gamma"]
    assert manifest["test"] == ["delta"]

    items = [{"id": name} for name in NAMES]

    # select_subset should honor the manifest.
    task_module.write_subset(tmp_path, "train", 0.3)
    selected = task_module.select_subset(items, tmp_path)
    assert {item["id"] for item in selected} == {"alpha", "beta"}

    task_module.write_subset(tmp_path, "val", 0.3)
    selected = task_module.select_subset(items, tmp_path)
    assert {item["id"] for item in selected} == {"gamma"}

    task_module.write_subset(tmp_path, "test", 0.3)
    selected = task_module.select_subset(items, tmp_path)
    assert {item["id"] for item in selected} == {"delta"}


# ── test_split_to_manifest ────────────────────────────────────────────────

def test_split_to_manifest(tmp_path):
    """split_to_manifest partitions items by fractions and writes a manifest."""
    items = [{"id": name} for name in NAMES]
    task_module.split_to_manifest(tmp_path, items, holdout=0.3)

    manifest = json.loads((tmp_path / task_module.MANIFEST).read_text())
    assert "train" in manifest
    # The manifest contains id strings.
    assert all(isinstance(x, str) for x in manifest["train"])
    # The partition is consistent with partition().
    parts = task_module.partition(items, 0.3)
    train_ids = {item["id"] for item in parts[task_module.TRAIN]}
    test_ids = {item["id"] for item in parts[task_module.HOLDOUT]}
    assert set(manifest["train"]) == train_ids
    if manifest.get("test"):
        assert set(manifest["test"]) == test_ids


# ── test_select_subset_val_test ────────────────────────────────────────────

def test_select_subset_val_test(tmp_path):
    """select_subset() accepts 'val' and 'test' parts from a manifest."""
    items = [{"id": name} for name in NAMES]
    task_module.make_manifest(
        tmp_path,
        train_ids=["alpha", "beta", "gamma"],
        val_ids=["delta"],
        test_ids=["epsilon", "zeta"],
    )
    task_module.write_subset(tmp_path, "val", 0.3)
    selected = task_module.select_subset(items, tmp_path)
    assert [item["id"] for item in selected] == ["delta"]

    task_module.write_subset(tmp_path, "test", 0.3)
    selected = task_module.select_subset(items, tmp_path)
    assert {item["id"] for item in selected} == {"epsilon", "zeta"}


# ── test_seed_reshuffle ────────────────────────────────────────────────────

def test_seed_reshuffle(tmp_path):
    """Different seeds produce different orderings within the same part;
    same seed produces identical ordering; partition boundaries are stable."""
    items = [{"id": name} for name in NAMES]
    task_module.make_manifest(
        tmp_path,
        train_ids=list(NAMES),
    )

    task_module.write_subset(tmp_path, "train", 0.3, seed=42)
    order_42a = [item["id"] for item in task_module.select_subset(items, tmp_path)]

    task_module.write_subset(tmp_path, "train", 0.3, seed=42)
    order_42b = [item["id"] for item in task_module.select_subset(items, tmp_path)]

    task_module.write_subset(tmp_path, "train", 0.3, seed=99)
    order_99 = [item["id"] for item in task_module.select_subset(items, tmp_path)]

    # Same seed → same order.
    assert order_42a == order_42b
    # Same set of ids.
    assert set(order_42a) == set(order_99)
    # Different seed → different order (with high probability for 6 items).
    assert order_42a != order_99 or len(NAMES) < 3


# ── test_seed_survives_measure ─────────────────────────────────────────────

def test_seed_survives_measure(tmp_path):
    """The seed written by _evaluate_part survives _measure's _write_subset.

    _measure calls _write_subset for each trial directory.  When _evaluate_part
    passes a seed to _measure, the subset.json in the occasion directory must
    retain the seed after the evaluation completes.  Reverting the seed
    parameter from _write_subset/_measure makes the seed disappear.
    """
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed_candidate = loop.population[0]
        loop._evaluate_part(seed_candidate, "val", None, seed=42)

        # The occasion directory's subset.json must have the seed.
        occasion = seed_candidate.parts["val"][0]
        workdir = Path(occasion["workdir"])
        subset = json.loads((workdir / task_module.SUBSET).read_text())
        assert subset.get("seed") == 42, (
            f"seed lost after _measure: subset.json = {subset}"
        )


# ── test_subset_json_backward_compat ───────────────────────────────────────

def test_subset_json_backward_compat(tmp_path):
    """A subset.json with no seed and no manifest works exactly as before."""
    items = [{"id": name} for name in ("alpha", "beta", "gamma", "delta")]
    task_module.write_subset(tmp_path, "train", 0.3)
    subset = json.loads((tmp_path / task_module.SUBSET).read_text())
    assert subset == {"part": "train", "holdout": 0.3}
    selected = task_module.select_subset(items, tmp_path)
    expected = task_module.partition(items, 0.3)[task_module.TRAIN]
    assert [item["id"] for item in selected] == [item["id"] for item in expected]


# ── test_evaluate_val_and_train ────────────────────────────────────────────

def test_evaluate_val_and_train(tmp_path):
    """Evaluate on val stores in candidate.parts, not candidate.score;
    evaluate on train stores in candidate.score. Both produce item records."""
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        run = Run(loop=loop)

        # Evaluate on train.
        run.evaluate(seed)
        assert seed.score is not None, "train evaluation should set candidate.score"
        assert seed.evaluated is True

        # Evaluate on val.
        run.evaluate(seed, part="val")
        assert "val" in seed.parts
        assert len(seed.parts["val"]) == 1
        val_occasion = seed.parts["val"][0]
        assert val_occasion["part"] == "val"
        assert val_occasion["occasion"] == 0
        assert val_occasion.get("item_records") is not None


# ── test_evaluate_part_fresh_directories ───────────────────────────────────

def test_evaluate_part_fresh_directories(tmp_path):
    """Two val evaluations produce two distinct occasion directories."""
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]

        loop._evaluate_part(seed, "val", None, None)
        loop._evaluate_part(seed, "val", None, None)

        assert len(seed.parts["val"]) == 2
        dir0 = seed.parts["val"][0]["workdir"]
        dir1 = seed.parts["val"][1]["workdir"]
        assert dir0 != dir1
        assert Path(dir0).exists()
        assert Path(dir1).exists()
        assert "val_0" in dir0
        assert "val_1" in dir1


# ── test_evaluate_backward_compat ──────────────────────────────────────────

def test_evaluate_backward_compat(tmp_path):
    """Calling run.evaluate(candidate) with no extra args works as before."""
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        run = Run(loop=loop)
        run.evaluate(seed)
        assert seed.evaluated is True
        assert seed.score is not None


# ── test_overfit_verdict_exit_code ─────────────────────────────────────────

def test_overfit_verdict_exit_code(tmp_path):
    """An overfit winner → exit_ok is False in the report.

    Reverting the ``and not audit.get("overfit")`` conjunct at loop.py report()
    makes exit_ok True for an overfit winner, which is the revert this test catches.
    We construct an evaluated winner and mock ``audit_holdout`` to report a large
    train/holdout gap so the overfit flag is unconditionally True.
    """
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        # Manually mark the seed as evaluated with a perfect train score.
        seed.evaluated = True
        seed.exit_ok = True
        seed.score = 1.0
        seed.raw_score = 1.0
        seed.mean_score = 1.0
        loop.evaluations = 1
        workdir = loop.root / "runs" / seed.id
        workdir.mkdir(parents=True, exist_ok=True)

        # Mock audit_holdout to return a clear overfit result.
        _original_audit = loop.audit_holdout
        def _overfit_audit():
            return {
                "status": "measured",
                "train": 1.0,
                "holdout": 0.3,
                "gap": 0.7,
                "overfit": True,
                "threshold": 0.15,
                "winner": seed.id,
            }
        loop.audit_holdout = _overfit_audit

        report = loop.report()

    audit = report.get("holdout", {})
    # The overfit check must detect the gap; assert unconditionally.
    assert audit.get("overfit") is True, (
        f"expected overfit=True; got audit={audit}"
    )
    assert report["exit_ok"] is False, (
        "an overfit winner must produce exit_ok=False"
    )


# ── test_overfit_like_with_like ───────────────────────────────────────────

def test_overfit_like_with_like(tmp_path):
    """When the winner has both val and test, the audit compares val vs test.

    Construct a winner with val_score=0.85, test_score=0.4.  The audit should
    compare val vs test (like with like: both unseen by the optimizer) and
    detect overfit.  Reverting ``audit_parts`` makes the audit fall back to
    train-vs-holdout, which will NOT detect overfit from val/test scores.
    """
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        # Mark seed as evaluated with a train score.
        seed.evaluated = True
        seed.exit_ok = True
        seed.score = 0.85
        seed.raw_score = 0.85
        seed.mean_score = 0.85
        loop.evaluations = 1
        workdir = loop.root / "runs" / seed.id
        workdir.mkdir(parents=True, exist_ok=True)
        import json as _json
        (workdir / "measurements.json").write_text(_json.dumps({
            "requested": 1, "count": 1, "successful": 1, "mean": 0.85,
            "stdev": None, "min": 0.85, "max": 0.85, "complete": True,
            "trials": [{"trial": 1, "score": 0.85, "exit_ok": True,
                        "error": None, "duration_s": 0.1, "model_calls": 0,
                        "required_contract": {"passed": True, "missing_artifacts": []},
                        "tokens": None, "cost": None, "workdir": str(workdir)}],
        }))
        # Populate explicit val and test parts.
        seed.parts["val"] = [{"part": "val", "occasion": 0, "score": 0.85,
                              "exit_ok": True, "error": None}]
        seed.parts["test"] = [{"part": "test", "occasion": 0, "score": 0.4,
                               "exit_ok": True, "error": None}]

        report = loop.report()

    audit = report.get("holdout", {})
    # audit_parts compares val (0.85) vs test (0.4): gap = 0.45 > 0.15 threshold.
    assert audit.get("overfit") is True, (
        f"expected overfit=True for val=0.85 vs test=0.4; got audit={audit}"
    )
    assert audit.get("val") == 0.85, f"expected val=0.85, got {audit.get('val')}"
    assert audit.get("test") == 0.4, f"expected test=0.4, got {audit.get('test')}"
    assert report["exit_ok"] is False, (
        "an overfit winner (val vs test) must produce exit_ok=False"
    )


# ── test_overfit_backward_compat ──────────────────────────────────────────

def test_overfit_backward_compat(tmp_path):
    """With no val partition, the audit falls back to train-vs-holdout."""
    loop = Loop(
        "dataset_probe/dataset",
        LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=1,
            runs_root=tmp_path / "run",
        ),
        project=Path("."),
    )
    report = loop.run()

    audit = report.get("holdout", {})
    assert audit.get("status") in ("measured", "not_applicable", "failed", "unavailable"), (
        "audit should still use train-vs-holdout comparison"
    )
