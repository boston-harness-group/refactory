"""--seed-results: a run reads its seed's first evaluation per part from an earlier run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import seed_cache
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.outerloop import Run

from test_parts import MODE, PartsProbe, _patch_mode

PARTS = {"train": ["alpha", "beta", "gamma", "delta"], "val": ["epsilon"], "test": ["zeta"]}


def _loop(root: Path, manifest: Path, **config) -> Loop:
    return Loop(MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1,
                                 runs_root=root, parts_manifest=str(manifest), **config),
                project=Path("."))


def _manifest(tmp_path: Path, parts=PARTS, name="parts.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(parts))
    return path


def _source_run(tmp_path: Path) -> Path:
    """A run whose seed was evaluated on train and val; returns its export."""
    loop = _loop(tmp_path / "source", _manifest(tmp_path))
    loop.init()
    run = Run(_loop=loop)
    run.evaluate(loop.population[0])
    run.evaluate(loop.population[0], part="val")
    cache = seed_cache.export(tmp_path / "source", project=Path("."))
    # Every item passes for real, so mark the cached result to tell it apart.
    cache["parts"]["train"]["result"]["score"] = 0.25
    cache["parts"]["val"]["result"]["score"] = 0.5
    path = tmp_path / "seed-results.json"
    path.write_text(json.dumps(cache))
    return path


def _events(root: Path, name: str) -> list[dict]:
    lines = (root / "journal.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r.get("event") == name]


def test_seed_results_stand_in_for_the_first_evaluation_per_part(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)

        run.evaluate(seed)
        assert seed.raw_score == 0.25
        assert loop.rollouts == 0
        assert {r["id"] for r in loop._train_item_records[seed.id]} == set(PARTS["train"])
        # The source's pass files are copied in, as if the evaluation ran here.
        assert (tmp_path / "reuse" / "runs" / seed.id / "passes").is_dir()

        run.evaluate(seed, part="val")
        assert loop._part_results[seed.id]["val"][0]["score"] == 0.5
        assert loop.rollouts == 0

        # A second occasion is a fresh measurement, and it runs and is charged.
        run.evaluate(seed, part="val")
        assert loop._part_results[seed.id]["val"][1]["score"] == 1.0
        assert loop.rollouts == 1
        # No test result was cached, so test runs for real.
        run.evaluate(seed, part="test")
        assert loop.rollouts == 2

        hits = _events(tmp_path / "reuse", "seed_cache_hit")
        assert [hit["part"] for hit in hits] == ["train", "val"]


def test_a_different_score_context_stops_the_run(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache), dsh_timeout_s=60)
        with pytest.raises(LoopError, match="dsh_timeout_s"):
            loop.init()


def test_the_inactivity_limit_is_part_of_the_context(tmp_path, monkeypatch):
    from refactory_dsh import model

    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        monkeypatch.setattr(model, "DEFAULT_INACTIVITY_S", 0)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        with pytest.raises(LoopError, match="model_inactivity_s"):
            loop.init()


def test_search_settings_are_not_part_of_the_context(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache),
                     max_rollouts=7, strategy_options={"group_size": 8})
        loop.config.epochs = 5
        loop.init()
        Run(_loop=loop).evaluate(loop.population[0])
        assert loop.population[0].raw_score == 0.25


def test_a_different_item_set_is_evaluated(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        other = _manifest(tmp_path, {**PARTS, "train": ["alpha", "beta"]}, name="other.json")
        loop = _loop(tmp_path / "reuse", other, seed_results=str(cache))
        loop.init()
        Run(_loop=loop).evaluate(loop.population[0])
        assert loop.population[0].raw_score == 1.0
        assert loop.rollouts == 2
        assert not _events(tmp_path / "reuse", "seed_cache_hit")


def test_a_train_minibatch_is_evaluated(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        loop.init()
        Run(_loop=loop).evaluate(loop.population[0], items=["alpha"])
        assert loop.rollouts == 1


def test_resume_remembers_which_parts_were_served(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        manifest = _manifest(tmp_path)
        first = _loop(tmp_path / "reuse", manifest, seed_results=str(cache))
        first.init()
        Run(_loop=first).evaluate(first.population[0], part="val")

        resumed = _loop(tmp_path / "reuse", manifest, seed_results=str(cache), resume=True)
        resumed.init()
        Run(_loop=resumed).evaluate(resumed.population[0], part="val")
        assert resumed._part_results[resumed.population[0].id]["val"][-1]["score"] == 1.0
        assert resumed.rollouts == 1


def test_export_of_a_run_without_a_recorded_context_needs_the_inactivity_limit(tmp_path):
    with _patch_mode(PartsProbe):
        _source_run(tmp_path)
        source = tmp_path / "source"
        recorded = json.loads((source / seed_cache.CONTEXT_FILE).read_text())
        (source / seed_cache.CONTEXT_FILE).unlink()
        with pytest.raises(seed_cache.SeedCacheError, match="--inactivity-s"):
            seed_cache.export(source, project=Path("."))
        rebuilt = seed_cache.export(source, project=Path("."),
                                    inactivity_s=recorded["model_inactivity_s"])
        assert rebuilt["score_context"] == recorded
        assert set(rebuilt["parts"]) == {"train", "val"}
