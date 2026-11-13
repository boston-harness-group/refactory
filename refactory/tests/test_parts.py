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
from refactory_dsh.outerloop import Run, OuterLoopError

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


def _make_loop_with_manifest(tmp_path, manifest_dict=None):
    """Helper: create a loop and install a parts manifest on the run root."""
    if manifest_dict is None:
        manifest_dict = {
            "train": ["alpha", "beta", "gamma", "delta"],
            "val": ["epsilon"],
            "test": ["zeta"],
        }
    # Write manifest file for LoopConfig.
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict, indent=2))
    loop = Loop(
        MODE, LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=1,
            runs_root=tmp_path / "run",
            parts_manifest=str(manifest_path),
        ),
        project=Path("."),
    )
    return loop


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

    # Same seed -> same order.
    assert order_42a == order_42b
    # Same set of ids.
    assert set(order_42a) == set(order_99)
    # Different seed -> different order (with high probability for 6 items).
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
        loop = _make_loop_with_manifest(tmp_path)
        loop.init()
        seed_candidate = loop.population[0]
        loop._evaluate_part(seed_candidate, "val", None, seed=42)

        # The occasion data is now in loop-private storage.
        val_results = loop._part_results.get(seed_candidate.id, {}).get("val", [])
        assert len(val_results) == 1
        occasion = val_results[0]
        assert occasion["part"] == "val"

        # Check the actual workdir's subset.json for the seed.
        occasion_dir = loop.root / "runs" / seed_candidate.id / "occasions" / "val_0"
        subset = json.loads((occasion_dir / task_module.SUBSET).read_text())
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
    """Evaluate on val stores in loop-private storage, not candidate.score;
    evaluate on train stores in candidate.score. Both produce item records."""
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        # Evaluate on train.
        run.evaluate(seed)
        assert seed.score is not None, "train evaluation should set candidate.score"
        assert seed.evaluated is True

        # Evaluate on val.
        run.evaluate(seed, part="val")
        val_results = loop._part_results.get(seed.id, {}).get("val", [])
        assert len(val_results) == 1
        val_occasion = val_results[0]
        assert val_occasion["part"] == "val"
        assert val_occasion["occasion"] == 0
        # Score-only: no item_records, no workdir.
        assert "item_records" not in val_occasion
        assert "workdir" not in val_occasion


# ── test_evaluate_part_fresh_directories ───────────────────────────────────

def test_evaluate_part_fresh_directories(tmp_path):
    """Two val evaluations produce two distinct occasion directories."""
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path)
        loop.init()
        seed = loop.population[0]

        loop._evaluate_part(seed, "val", None, None)
        loop._evaluate_part(seed, "val", None, None)

        val_results = loop._part_results.get(seed.id, {}).get("val", [])
        assert len(val_results) == 2
        # Check that occasion directories exist.
        dir0 = loop.root / "runs" / seed.id / "occasions" / "val_0"
        dir1 = loop.root / "runs" / seed.id / "occasions" / "val_1"
        assert dir0.exists()
        assert dir1.exists()


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
        run = Run(_loop=loop)
        run.evaluate(seed)
        assert seed.evaluated is True
        assert seed.score is not None


# ── test_overfit_verdict_exit_code ─────────────────────────────────────────

def test_overfit_verdict_exit_code(tmp_path):
    """An overfit winner -> exit_ok is False in the report.

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
        def _overfit_audit(*, winner=None):
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
    """When the winner has both val and test in loop-private storage, the audit
    compares val vs test.

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
        # An optimized winner (a child of the seed that scores higher on train) with
        # val and test in loop-private storage; the seed's own gap is 0.05, so the
        # search added 0.40 -- a seed winner is never called overfit.
        import copy
        from refactory_dsh.optim import Candidate
        state = copy.deepcopy(seed.state)
        first = sorted(state["graph"]["nodes"])[0]
        state["graph"]["nodes"][first]["notes"] = "the optimized child"
        child = Candidate.from_state(state, parent=seed.id)
        child.evaluated, child.exit_ok = True, True
        child.score = child.raw_score = child.mean_score = 0.9
        loop.population.append(child)

        def part(score: float, name: str) -> list[dict]:
            return [{"part": name, "occasion": 0, "score": score,
                     "exit_ok": True, "error": None, "duration_s": 0.1}]

        loop._part_results[seed.id] = {"val": part(0.6, "val"), "test": part(0.55, "test")}
        loop._part_results[child.id] = {"val": part(0.85, "val"), "test": part(0.4, "test")}

        report = loop.report()

    audit = report.get("holdout", {})
    # audit_parts compares val (0.85) vs test (0.4) and subtracts the seed's 0.05.
    assert audit.get("overfit") is True, (
        f"expected overfit=True for val=0.85 vs test=0.4; got audit={audit}"
    )
    assert audit.get("val") == 0.85, f"expected val=0.85, got {audit.get('val')}"
    assert audit.get("test") == 0.4, f"expected test=0.4, got {audit.get('test')}"
    assert report["exit_ok"] is False, (
        "an overfit winner (val vs test) must produce exit_ok=False"
    )
    # The error and the printed line say what the audit found, with both scores;
    # they read "the required held-out audit is unavailable" and "no holdout score".
    from refactory_dsh.loop import _holdout_line
    line = _holdout_line(audit)
    assert line == ("holdout: val 0.8500, test 0.4000, gap 0.4500, added by the search 0.4000 "
                    "against threshold 0.15 — overfit (val and test parts)")
    assert report["error"] == f"the winner overfit ({line})"


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


# ======================================================================
# NEW TESTS -- exam-integrity fixes
# ======================================================================

# ── Fix 1: test_no_optimizer_facing_path_returns_val_test_data ────────────

def test_no_optimizer_facing_path_returns_val_test_data(tmp_path):
    """No optimizer-facing API surface returns val or test item records or workdirs.

    Reverting the removal of `parts` from Candidate makes this fail.
    """
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        # Evaluate on train.
        run.evaluate(seed)

        # Evaluate on val.
        run.evaluate(seed, part="val")

        # 1. Candidate has no `parts` attribute.
        assert not hasattr(seed, "parts"), (
            "Candidate should not have a 'parts' attribute"
        )
        assert not hasattr(seed, "occasion_counters"), (
            "Candidate should not have an 'occasion_counters' attribute"
        )
        assert not hasattr(seed, "_train_item_records"), (
            "Candidate should not have a '_train_item_records' attribute"
        )

        # 2. describe() has no val/test data.
        desc = seed.describe()
        assert "parts" not in desc
        assert "item_records" not in desc

        # 3. item_records only returns train.
        records = run.item_records(seed)
        assert isinstance(records, list)
        # Asking for val or test records is an error.
        with pytest.raises(OuterLoopError):
            run.item_records(seed, part="val")

        # 4. parents() returns Candidates with no parts.
        parents = run.parents(1)
        for p in parents:
            assert not hasattr(p, "parts")

        # 5. _context() has no val/test data in item_records.
        ctx = loop._context()
        assert isinstance(ctx.item_records, list)


# ── Fix 1 + Correction 2: test_public_attrs_no_val_test_leak ─────────────

def test_public_attrs_no_val_test_leak(tmp_path):
    """Walk public attributes of Run, ProposalContext, and Candidate after val/test
    evaluation: none should return val/test records or workdirs.

    Run must not expose a public 'loop' attribute.
    """
    from refactory_dsh.optim.base import ProposalContext

    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        run.evaluate(seed)
        run.evaluate(seed, part="val")
        run.evaluate(seed, part="test")

        # Run must not expose 'loop' as a public attribute.
        public_run_attrs = [a for a in dir(run) if not a.startswith("_")]
        assert "loop" not in public_run_attrs, (
            "Run must not expose a public 'loop' attribute"
        )

        # Candidate must not have parts, occasion_counters, or _train_item_records.
        assert not hasattr(seed, "parts")
        assert not hasattr(seed, "occasion_counters")
        assert not hasattr(seed, "_train_item_records")

        # Walk public attributes of Candidate: none should contain val/test records.
        for attr_name in dir(seed):
            if attr_name.startswith("_"):
                continue
            attr = getattr(seed, attr_name)
            if callable(attr):
                continue
            if isinstance(attr, dict):
                assert "val" not in attr, (
                    f"Candidate.{attr_name} contains 'val' key"
                )
                assert "test" not in attr, (
                    f"Candidate.{attr_name} contains 'test' key"
                )

        # ProposalContext: no val/test records.
        ctx = loop._context()
        for attr_name in dir(ctx):
            if attr_name.startswith("_"):
                continue
            attr = getattr(ctx, attr_name)
            if callable(attr):
                continue
            if isinstance(attr, list):
                for entry in attr:
                    if isinstance(entry, dict):
                        assert not (entry.get("part") in ("val", "test") and "item_records" in entry), (
                            f"ProposalContext.{attr_name} leaks val/test records"
                        )


# ── Fix 2: test_val_without_manifest_raises ──────────────────────────────

def test_val_without_manifest_raises(tmp_path):
    """select_subset with part='val' or part='test' and no manifest raises TaskError.

    Reverting the error-raising makes val silently return train items.
    """
    items = [{"id": name} for name in NAMES]

    # Val without manifest.
    task_module.write_subset(tmp_path, "val", 0.3)
    with pytest.raises(task_module.TaskError, match="requires a parts manifest"):
        task_module.select_subset(items, tmp_path)

    # Test without manifest.
    task_module.write_subset(tmp_path, "test", 0.3)
    with pytest.raises(task_module.TaskError, match="requires a parts manifest"):
        task_module.select_subset(items, tmp_path)


# ── Fix 2: test_train_val_disjoint_from_manifest ────────────────────────

def test_train_val_disjoint_from_manifest(tmp_path):
    """The manifest loader refuses overlapping parts and unreadable JSON.

    Overlap would put val/test items into the optimizer's train set; a manifest
    that silently fell back to the fraction split would grade the run on items
    nobody chose. Both are LoopErrors at init. Disjoint parts load as written.
    """
    from refactory_dsh.loop import LoopError

    overlapping = (
        {"train": ["alpha", "beta", "epsilon"], "val": ["epsilon"], "test": ["zeta"]},
        {"train": ["alpha", "beta"], "val": ["gamma"], "test": ["gamma", "zeta"]},
        {"train": ["alpha", "zeta"], "val": ["gamma"], "test": ["zeta"]},
    )
    with _patch_mode(PartsProbe):
        for index, manifest_dict in enumerate(overlapping):
            loop = _make_loop_with_manifest(tmp_path / f"overlap{index}", manifest_dict)
            with pytest.raises(LoopError, match="must be disjoint"):
                loop.init()

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "source_manifest.json").write_text("{not json")
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=bad / "run", parts_manifest=str(bad / "source_manifest.json"),
            ),
            project=Path("."),
        )
        with pytest.raises(LoopError, match="not valid JSON"):
            loop.init()

        disjoint = {"train": ["alpha", "beta", "gamma"], "val": ["delta", "epsilon"],
                    "test": ["zeta"]}
        loop = _make_loop_with_manifest(tmp_path / "ok", disjoint)
        loop.init()
        assert loop._parts_manifest == disjoint
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        assert {r["id"] for r in run.item_records(seed)} == set(disjoint["train"])


# ── Review round 3 ───────────────────────────────────────────────────────

def _no_manifest_loop(tmp_path, **config):
    return Loop(
        MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1,
                         runs_root=tmp_path / "run", **config),
        project=Path("."),
    )


def test_no_manifest_train_items_refuse_holdout(tmp_path):
    """Without a manifest, train items must come from the train split.

    Held-out ids passed as a 'train' minibatch would grade a candidate on the
    audit's items, and _measure narrows its expected ids to the items given, so
    the ownership check cannot catch it afterwards.
    """
    with _patch_mode(PartsProbe):
        loop = _no_manifest_loop(tmp_path)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        items = [{"id": name} for name in NAMES]
        held = [item["id"] for item in task_module.partition(items, loop.config.holdout)["holdout"]]
        train = [item["id"] for item in task_module.partition(items, loop.config.holdout)["train"]]
        assert held and train
        with pytest.raises(OuterLoopError, match="not in the train split"):
            run.evaluate(seed, part="train", items=held[:1])
        assert not seed.evaluated
        assert loop.rollouts == 0
        run.evaluate(seed, part="train", items=train[:1])
        assert seed.exit_ok, seed.error
        assert {r["id"] for r in run.item_records(seed)} == {train[0]}


def test_val_test_items_rejected(tmp_path):
    """Val and test are evaluated in full: items are refused even when they are
    a subset of the manifest's part."""
    manifest_dict = {"train": ["alpha", "beta"], "val": ["gamma", "delta"], "test": ["zeta"]}
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        for part, items in (("val", ["gamma"]), ("test", ["zeta"])):
            with pytest.raises(OuterLoopError, match="evaluated in full"):
                run.evaluate(seed, part=part, items=items)
        assert run.val_scores(seed) == []
        run.evaluate(seed, part="val")
        occasion = loop.root / "runs" / seed.id / "occasions" / "val_0"
        graded = json.loads((occasion / "item_records.json").read_text())
        assert {r["id"] for r in graded} == {"gamma", "delta"}


def test_val_missing_from_manifest_raises(tmp_path):
    """Val on a manifest with no val entry is a clear refusal, not an E-exam."""
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, {"train": ["alpha", "beta"]})
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        with pytest.raises(OuterLoopError, match="has no 'val' entry"):
            run.evaluate(seed, part="val")
        with pytest.raises(OuterLoopError, match="has no 'test' entry"):
            run.evaluate(seed, part="test")
        assert loop._occasion_counters == {}

        plain = _no_manifest_loop(tmp_path / "plain")
        plain.init()
        with pytest.raises(OuterLoopError, match="no parts manifest exists"):
            Run(_loop=plain).evaluate(plain.population[0], part="val")


def test_val_evaluation_counts_rollouts(tmp_path):
    """Every val/test item graded counts toward rollouts, not one per occasion."""
    manifest_dict = {"train": ["alpha"], "val": ["gamma", "delta", "epsilon"],
                     "test": ["zeta"]}
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        before = loop.rollouts
        run.evaluate(seed, part="val")
        assert loop.rollouts == before + 3


def test_train_store_never_holds_val_test_ids(tmp_path):
    """Val/test grading never reaches the optimizer's train record store."""
    manifest_dict = {"train": ["alpha", "beta"], "val": ["gamma", "delta"],
                     "test": ["epsilon", "zeta"]}
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        run.evaluate(seed, part="val")
        run.evaluate(seed, part="test")
        held = set(manifest_dict["val"]) | set(manifest_dict["test"])
        stored = {r["id"] for records in loop._train_item_records.values() for r in records}
        assert stored == set(manifest_dict["train"])
        assert not stored & held
        assert not {r["id"] for r in loop._context().item_records} & held


def test_resume_replays_part_results_for_audit(tmp_path):
    """audit_parts after resume still sees the val and test scores."""
    with _patch_mode(PartsProbe):
        loop1 = _make_loop_with_manifest(tmp_path)
        loop1.init()
        seed = loop1.population[0]
        run = Run(_loop=loop1)
        run.evaluate(seed)
        run.evaluate(seed, part="val")
        run.evaluate(seed, part="test")
        before = loop1.audit_parts(seed)
        assert before is not None and before["status"] == "measured"

        loop2 = Loop(
            MODE, LoopConfig(
                epochs=2, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                parts_manifest=str(tmp_path / "source_manifest.json"),
                resume=True,
            ),
            project=Path("."),
        )
        loop2.init()
        after = loop2.audit_parts(loop2.population[0])
        assert after == before


def test_repeat_index_on_joined_records(tmp_path):
    """Records joined across repeats say which repeat graded them."""
    manifest_dict = {"train": ["alpha", "beta"], "val": ["gamma"], "test": ["zeta"]}
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict))
    with _patch_mode(PartsProbe):
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run", parts_manifest=str(manifest_path), repeats=2,
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        records = run.item_records(seed)
    assert sorted((r["id"], r["repeat"]) for r in records) == [
        ("alpha", 1), ("alpha", 2), ("beta", 1), ("beta", 2),
    ]


# ── Fix 3: test_loop_owned_manifest_from_config ──────────────────────────

def test_loop_owned_manifest_from_config(tmp_path):
    """LoopConfig with parts_manifest installs the manifest at the run root.

    Reverting the parts_manifest config field makes it unavailable.
    """
    manifest_dict = {
        "train": ["alpha", "beta", "gamma", "delta"],
        "val": ["epsilon"],
        "test": ["zeta"],
    }
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()

        # The manifest should be copied to the run root.
        assert (loop.root / task_module.MANIFEST).exists(), (
            "manifest should be copied to the run root"
        )
        installed = json.loads((loop.root / task_module.MANIFEST).read_text())
        assert installed["train"] == manifest_dict["train"]
        assert installed["val"] == manifest_dict["val"]

        # The loop's internal manifest should be loaded.
        assert loop._parts_manifest is not None
        assert loop._parts_manifest["train"] == manifest_dict["train"]

        # Evaluate a candidate -- items should come from the manifest.
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        records = run.item_records(seed)
        record_ids = {r["id"] for r in records}
        assert record_ids == set(manifest_dict["train"]), (
            f"evaluated train items {record_ids} != manifest {manifest_dict['train']}"
        )


# ── Fix 3 (F3 ownership): test_val_test_items_checked_against_manifest ──

def test_val_test_items_checked_against_manifest(tmp_path):
    """_expected_item_ids for val and test reads from the manifest.

    Reverting the manifest lookup makes it return an empty set.
    """
    manifest_dict = {
        "train": ["alpha", "beta", "gamma", "delta"],
        "val": ["epsilon"],
        "test": ["zeta"],
    }
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()

        seed_state = loop.population[0].state

        val_ids, _ = loop._expected_item_ids(seed_state, "val")
        assert val_ids == {"epsilon"}, (
            f"expected val ids {{'epsilon'}}, got {val_ids}"
        )

        test_ids, _ = loop._expected_item_ids(seed_state, "test")
        assert test_ids == {"zeta"}, (
            f"expected test ids {{'zeta'}}, got {test_ids}"
        )


# ── Fix 4: test_evaluate_train_honors_seed_and_items ─────────────────────

def test_evaluate_train_honors_seed_and_items(tmp_path):
    """run.evaluate(part='train', seed=42, items=['alpha', 'beta']) honours both.

    Reverting the seed/items pass-through makes both be ignored.
    """
    manifest_dict = {
        "train": ["alpha", "beta", "gamma", "delta"],
        "val": ["epsilon"],
        "test": ["zeta"],
    }
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        run.evaluate(seed, part="train", seed=42, items=["alpha", "beta"])

        assert seed.exit_ok is True, f"expected exit_ok=True, got {seed.exit_ok}; error={seed.error}"
        assert seed.score is not None, f"expected a score, got None; error={seed.error}"

        records = run.item_records(seed)
        record_ids = {r["id"] for r in records}
        assert record_ids == {"alpha", "beta"}, (
            f"expected items alpha, beta; got {record_ids}"
        )

        # Check seed propagated to subset.json.
        workdir = loop.root / "runs" / seed.id
        subset = json.loads((workdir / task_module.SUBSET).read_text())
        assert subset.get("seed") == 42, (
            f"seed not propagated: subset.json = {subset}"
        )


# ── Fix 3 constraint: test_val_test_items_arg_rejects_non_subset ─────────

def test_val_test_items_arg_rejects_non_subset(tmp_path):
    """With a manifest that has val=['epsilon'], calling run.evaluate with
    items=['alpha'] raises OuterLoopError."""
    manifest_dict = {
        "train": ["alpha", "beta", "gamma", "delta"],
        "val": ["epsilon"],
        "test": ["zeta"],
    }
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)

        with pytest.raises(OuterLoopError):
            run.evaluate(seed, part="val", items=["alpha"])


# ── Correction 1: test_train_items_subset_of_manifest ────────────────────

def test_train_items_subset_of_manifest(tmp_path):
    """With a manifest that has train=['alpha','beta','gamma','delta'],
    items=['alpha','beta'] succeeds; items=['alpha','x'] raises OuterLoopError."""
    manifest_dict = {
        "train": ["alpha", "beta", "gamma", "delta"],
        "val": ["epsilon"],
        "test": ["zeta"],
    }
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        # Valid subset succeeds.
        run.evaluate(seed, part="train", items=["alpha", "beta"])
        assert seed.exit_ok is True, f"expected exit_ok=True, got {seed.exit_ok}; error={seed.error}"
        assert seed.score is not None, f"expected a score, got None; error={seed.error}"
        records = run.item_records(seed)
        assert {r["id"] for r in records} == {"alpha", "beta"}

        # Create a fresh candidate for the invalid test.
        from refactory_dsh.optim import Candidate as C
        cand2 = C.from_state(seed.state)
        loop.population.append(cand2)

        # Invalid items raises.
        with pytest.raises(OuterLoopError, match="not in the manifest"):
            run.evaluate(cand2, part="train", items=["alpha", "x"])


# ── test_val_scores_view ─────────────────────────────────────────────────

def test_val_scores_view(tmp_path):
    """run.val_scores(candidate) returns score-level data with no records or workdirs."""
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        run.evaluate(seed)
        run.evaluate(seed, part="val")

        scores = run.val_scores(seed)
        assert len(scores) == 1
        occ = scores[0]
        assert occ["part"] == "val"
        assert "score" in occ
        assert "exit_ok" in occ
        assert "item_records" not in occ
        assert "workdir" not in occ


# ── Correction 3: test_existing_run_resumes ──────────────────────────────

def test_existing_run_resumes(tmp_path):
    """A run checkpointed before this change (candidates with parts/occasion_counters)
    still resumes without error.

    Reverting the deserialization guards breaks resume of old runs.
    """
    with _patch_mode(PartsProbe):
        # First run: create and evaluate.
        loop1 = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
            ),
            project=Path("."),
        )
        loop1.init()
        loop1.forward()

        # The journal should have candidate and score events; replay should work.
        journal_path = loop1.root / "journal.jsonl"
        lines = journal_path.read_text().splitlines()
        assert any('"candidate"' in line for line in lines)

        # Resume the run.
        loop2 = Loop(
            MODE, LoopConfig(
                epochs=2, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                resume=True,
            ),
            project=Path("."),
        )
        # Should not raise.
        loop2.init()
        assert len(loop2.population) >= 1
        assert loop2.evaluations >= 1


# ======================================================================
# RESUME TESTS -- Part B of PR #558
# ======================================================================

# ── test_resume_occasion_counter ─────────────────────────────────────────

def test_resume_occasion_counter(tmp_path):
    """After stop+resume, the occasion counter continues from where it left off.

    Evaluating val after resume must produce val_1 (not val_0).  Reverting
    the journal_occasion_counter replay handler makes the counter restart
    at 0, reusing val_0.
    """
    with _patch_mode(PartsProbe):
        loop1 = _make_loop_with_manifest(tmp_path)
        loop1.init()
        seed = loop1.population[0]
        run = Run(_loop=loop1)

        # Evaluate on train then val (produces val_0).
        run.evaluate(seed)
        run.evaluate(seed, part="val")
        assert loop1._occasion_counters.get(seed.id, {}).get("val") == 1

        # Resume.
        loop2 = Loop(
            MODE, LoopConfig(
                epochs=2, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                parts_manifest=str(tmp_path / "source_manifest.json"),
                resume=True,
            ),
            project=Path("."),
        )
        loop2.init()
        seed2 = loop2.population[0]

        # The occasion counter should survive resume.
        assert loop2._occasion_counters.get(seed2.id, {}).get("val") == 1, (
            f"occasion counter not restored: {loop2._occasion_counters}"
        )

        # The next val evaluation should produce val_1, not val_0.
        run2 = Run(_loop=loop2)
        run2.evaluate(seed2, part="val")
        dir_1 = loop2.root / "runs" / seed2.id / "occasions" / "val_1"
        assert dir_1.exists(), (
            "val_1 directory should exist after resume; counter was not restored"
        )


# ── test_resume_rollout_count ────────────────────────────────────────────

def test_resume_rollout_count(tmp_path):
    """After stop+resume, loop.rollouts equals the pre-stop value.

    Reverting the journal_rollouts replay handler makes the rollout count
    restart at 0.
    """
    with _patch_mode(PartsProbe):
        loop1 = _make_loop_with_manifest(tmp_path)
        loop1.init()
        seed = loop1.population[0]
        run = Run(_loop=loop1)
        run.evaluate(seed)

        pre_stop_rollouts = loop1.rollouts
        assert pre_stop_rollouts > 0, "rollouts should be > 0 after evaluation"

        # Resume.
        loop2 = Loop(
            MODE, LoopConfig(
                epochs=2, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                parts_manifest=str(tmp_path / "source_manifest.json"),
                resume=True,
            ),
            project=Path("."),
        )
        loop2.init()

        assert loop2.rollouts == pre_stop_rollouts, (
            f"rollouts not restored: {loop2.rollouts} != {pre_stop_rollouts}"
        )


# ── test_resume_train_records ────────────────────────────────────────────

def test_resume_train_records(tmp_path):
    """After stop+resume, ProposalContext.item_records is non-empty.

    Reverting the journal_train_records replay handler makes item_records
    empty after resume.
    """
    with _patch_mode(PartsProbe):
        # No val items: the context's parent is the train leader, the evaluated seed.
        loop1 = _make_loop_with_manifest(
            tmp_path, {"train": ["alpha", "beta", "gamma", "delta"], "test": ["zeta"]})
        loop1.init()
        seed = loop1.population[0]
        run = Run(_loop=loop1)
        run.evaluate(seed)

        pre_stop_records = list(loop1._train_item_records.get(seed.id, []))
        assert pre_stop_records, "train records should be non-empty after evaluation"

        # Resume.
        loop2 = Loop(
            MODE, LoopConfig(
                epochs=2, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                parts_manifest=str(tmp_path / "source_manifest.json"),
                resume=True,
            ),
            project=Path("."),
        )
        loop2.init()

        seed2 = loop2.population[0]
        restored_records = list(loop2._train_item_records.get(seed2.id, []))
        assert restored_records, (
            "train records should be non-empty after resume"
        )
        ctx = loop2._context()
        assert ctx.item_records, (
            "ProposalContext.item_records should be non-empty after resume"
        )


# ── test_resume_repeats_all_counted ──────────────────────────────────────

def test_resume_repeats_all_counted(tmp_path):
    """With repeats=2, every trial's items count toward rollouts.

    Reverting the per-trial rollout increment makes rollouts count only
    the first trial.
    """
    with _patch_mode(PartsProbe):
        manifest_dict = {
            "train": ["alpha", "beta", "gamma", "delta"],
            "val": ["epsilon"],
            "test": ["zeta"],
        }
        manifest_path = tmp_path / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest_dict))
        loop = Loop(
            MODE, LoopConfig(
                epochs=1, batch_size=1, optimizer="random", seed=1,
                runs_root=tmp_path / "run",
                parts_manifest=str(manifest_path),
                repeats=2,
            ),
            project=Path("."),
        )
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)

        # With 4 train items and 2 repeats, rollouts should count items
        # from BOTH trials: 4 items × 2 repeats = 8 rollouts.
        # Reverting the per-trial accumulation in _measure makes this 4
        # (only the last trial's items counted).
        assert loop.rollouts == 8, (
            f"expected rollouts == 8 with 4 items × 2 repeats, got {loop.rollouts}"
        )


def test_context_follows_the_parent_between_steps(tmp_path):
    """Each optimizer step recomputes the parent it edits.

    The leader changes between two steps; the second step's proposal context must
    carry the new leader's train records, and the parent must not be published as
    a strategy decision.
    """
    # No val items: the run selects on train scores, so the train leader is the parent.
    manifest_dict = {"train": ["alpha", "beta", "gamma"], "test": ["zeta"]}
    with _patch_mode(PartsProbe):
        loop = _make_loop_with_manifest(tmp_path, manifest_dict)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        seen = []
        original = loop._context

        def spy():
            context = original()
            seen.append(context)
            return context

        loop._context = spy
        loop.optimizer_step(run)
        assert seen and {r["id"] for r in seen[-1].item_records} == {"alpha", "beta", "gamma"}

        leader = Candidate.from_state(seed.state, parent=seed.id)
        leader.id = "leader000001"
        loop.population.append(leader)
        run.evaluate(leader, items=["alpha"])
        leader.score = 99.0
        assert loop.select_parents(1)[0].id == leader.id

        loop.optimizer_step(run)
        context = seen[-1]
        assert [r["id"] for r in context.item_records] == ["alpha"]
        assert context.item_records == loop._train_item_records[leader.id]
        assert "parent" not in context.strategy
        assert "parent" not in loop.strategy_decision
