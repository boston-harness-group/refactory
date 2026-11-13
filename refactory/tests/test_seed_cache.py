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
    config = {"epochs": 1, "batch_size": 1, "optimizer": "random", "seed": 1, **config}
    return Loop(MODE, LoopConfig(runs_root=root, parts_manifest=str(manifest), **config),
                project=Path("."))


def _manifest(tmp_path: Path, parts=PARTS, name="parts.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(parts))
    return path


def _source_run(tmp_path: Path, **config) -> Path:
    """A run whose seed was evaluated on train and val; returns its export."""
    loop = _loop(tmp_path / "source", _manifest(tmp_path), **config)
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


def test_a_different_val_item_set_stops_the_run(tmp_path):
    """A val part whose items differ from the cached ones is a cache that can never
    apply there: the run stops naming the part, instead of quietly measuring."""
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        other = _manifest(tmp_path, {**PARTS, "train": ["alpha", "beta", "gamma"], "val": ["delta"]},
                          name="other.json")
        loop = _loop(tmp_path / "reuse", other, seed_results=str(cache))
        with pytest.raises(LoopError, match=r"items of val \(1 cached, 1 in this run, 2 differ"):
            loop.init()


def test_other_train_items_are_measured_and_val_still_reused(tmp_path):
    """Train items other than the cached ones (a minibatch against the whole part)
    are measured and journaled as a miss; val is still served."""
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        other = _manifest(tmp_path, {**PARTS, "train": ["alpha", "beta"]}, name="other.json")
        loop = _loop(tmp_path / "reuse", other, seed_results=str(cache))
        loop.init()
        seed = loop.population[0]
        Run(_loop=loop).evaluate(seed)
        Run(_loop=loop).evaluate(seed, part="val")
        assert loop.rollouts == 2
        misses = _events(tmp_path / "reuse", "seed_cache_miss")
        assert [(m["part"], m["reason"]) for m in misses] == [("train", "other items")]
        assert [h["part"] for h in _events(tmp_path / "reuse", "seed_cache_hit")] == ["val"]


def test_another_seed_stops_the_run(tmp_path):
    """A run whose seed is not the cache's candidate (other starting knobs, another
    mode state) would never be served: it stops naming the candidate."""
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        data = json.loads(cache.read_text())
        data["candidate"] = "000000000000"
        cache.write_text(json.dumps(data))
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        with pytest.raises(LoopError, match="candidate"):
            loop.init()


def test_a_train_minibatch_is_evaluated(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        loop.init()
        Run(_loop=loop).evaluate(loop.population[0], items=["alpha"])
        assert loop.rollouts == 1
        # The miss is journaled once, so the run says why the cache was not used.
        Run(_loop=loop).evaluate(loop.population[0], items=["beta"])
        misses = _events(tmp_path / "reuse", "seed_cache_miss")
        assert [(m["part"], m["reason"]) for m in misses] == [("train", "other items")]


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


def test_repeats_export_and_reuse_every_trial(tmp_path):
    """With repeats 3 every trial is exported and served (the cache was silently
    unused for any repeats above 1, and the export collapsed the trials)."""
    with _patch_mode(PartsProbe):
        manifest = _manifest(tmp_path)
        source = _loop(tmp_path / "source", manifest, repeats=3)
        source.init()
        Run(_loop=source).evaluate(source.population[0])
        Run(_loop=source).evaluate(source.population[0], part="val")
        cache = seed_cache.export(tmp_path / "source", project=Path("."))
        assert sorted(cache["parts"]["train"]["items"]) == sorted(PARTS["train"])
        assert [len(t["item_records"]) for t in cache["parts"]["train"]["trials"]] == [4, 4, 4]
        assert len(cache["parts"]["val"]["trials"]) == 3
        path = tmp_path / "seed-results.json"
        path.write_text(json.dumps(cache))

        loop = _loop(tmp_path / "reuse", manifest, repeats=3, seed_results=str(path))
        loop.init()
        seed = loop.population[0]
        Run(_loop=loop).evaluate(seed)
        Run(_loop=loop).evaluate(seed, part="val")
        assert loop.rollouts == 0
        assert seed.measurements["count"] == 3 and seed.measurements["cached_from"]
        records = loop._train_item_records[seed.id]
        assert sorted({r["repeat"] for r in records}) == [1, 2, 3] and len(records) == 12
        # The later trials' run directories come along, as a live run keeps them.
        assert (tmp_path / "reuse" / "runs" / seed.id / "trials" / "3").is_dir()
        hits = _events(tmp_path / "reuse", "seed_cache_hit")
        assert [(h["part"], h["trials"]) for h in hits] == [("train", 3), ("val", 3)]


def test_a_cache_from_other_repeats_stops_the_run(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), repeats=3, seed_results=str(cache))
        with pytest.raises(LoopError, match="repeats"):
            loop.init()


# ── end to end, through the command line and a profile ──────────────────────

from refactory_dsh import loop as loop_module, profiles  # noqa: E402


def _main(tmp_path: Path, name: str, *argv: str, manifest: Path | None = None) -> tuple[int, Path]:
    root = tmp_path / name
    code = loop_module.main([MODE, "--project", str(tmp_path), "--runs-root", str(root),
                             "--parts", str(manifest or tmp_path / "parts.json"), "--holdout", "0",
                             "--seed", "3", *argv])
    return code, root


@pytest.fixture
def exported(tmp_path, monkeypatch):
    """Run A with the validation gate and three repeats, and its exported seed results."""
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    _manifest(tmp_path)
    with _patch_mode(PartsProbe):
        code, root = _main(tmp_path, "A", "--optimizer", "random",
                           "--epochs", "1", "--batch-size", "1", "--repeats", "3")
        assert code == 0
        out = tmp_path / "seed-results.json"
        assert seed_cache.main(["export", str(root), str(out), "--project", str(tmp_path)]) == 0
        yield out


@pytest.mark.parametrize("search", [
    ["--optimizer", "hill_climb", "--epochs", "2"],
    ["--optimizer", "random", "--epochs", "2", "--batch-size", "2"],
    ["--optimizer", "random", "--strategy", "tree_search", "--epochs", "1"],
])
def test_runs_with_other_search_settings_reuse_the_seed_and_say_so(tmp_path, exported, search):
    with _patch_mode(PartsProbe):
        code, root = _main(tmp_path, "B", *search, "--repeats", "3", "--seed-results", str(exported))
    assert code == 0
    hits = _events(root, "seed_cache_hit")
    assert hits and hits[0]["part"] == "train" and hits[0]["trials"] == 3
    seed = hits[0]["id"]
    candidate = json.loads((root / "runs" / seed / "candidate.json").read_text())
    assert candidate["measurements"]["cached_from"].endswith("#train")
    report = json.loads((root / "report.json").read_text())
    assert report["seed_results"]["parts_reused"] == sorted(h["part"] for h in hits)
    assert report["seed_results"]["source"] == str(tmp_path / "A")
    assert "was reused from" in (root / "report.md").read_text()


def test_a_profile_carries_the_seed_results(tmp_path, exported):
    profiles.save(tmp_path, "reuse", {"settings": {
        "optimizer": "random", "epochs": 1, "batch_size": 1, "repeats": 3,
        "parts_manifest": str(tmp_path / "parts.json"), "seed_results": str(exported)}})
    with _patch_mode(PartsProbe):
        code, root = _main(tmp_path, "B", "--profile", "reuse")
    assert code == 0
    assert {h["part"] for h in _events(root, "seed_cache_hit")} == {"train", "val", "test"}


def test_another_item_set_or_evaluator_is_refused_naming_it(tmp_path, exported, capsys):
    other = _manifest(tmp_path, {**PARTS, "train": ["alpha", "beta", "gamma"], "val": ["delta"]},
                      name="other.json")
    with _patch_mode(PartsProbe):
        code, root = _main(tmp_path, "C1", "--optimizer", "random", "--repeats", "3",
                           "--seed-results", str(exported), manifest=other)
        assert code == 2
        assert "items of val" in capsys.readouterr().err
        code, _ = _main(tmp_path, "C2", "--optimizer", "random", "--repeats", "3",
                        "--evaluator", "dsh", "--seed-results", str(exported))
        assert code == 2
        assert "different score context (evaluator)" in capsys.readouterr().err


def test_an_edited_mode_file_is_refused_naming_it(tmp_path, monkeypatch, capsys):
    """The mode's own file is in the score context: editing it refuses the cache."""
    import importlib
    import sys

    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    module_dir = tmp_path / "modes"
    module_dir.mkdir()
    source = module_dir / "edited_probe.py"
    source.write_text("from test_parts import PartsProbe\n\n\nclass EditedProbe(PartsProbe):\n"
                      "    name = 'edited_probe'\n")
    monkeypatch.syspath_prepend(str(module_dir))
    mode_class = importlib.import_module("edited_probe").EditedProbe
    _manifest(tmp_path)
    try:
        with _patch_mode(mode_class):
            code, root = _main(tmp_path, "A", "--optimizer", "random", "--epochs", "1")
            assert code == 0
            out = tmp_path / "seed-results.json"
            assert seed_cache.main(["export", str(root), str(out), "--project", str(tmp_path)]) == 0
            source.write_text(source.read_text() + "# an edit\n")
            code, _ = _main(tmp_path, "C", "--optimizer", "random", "--epochs", "1",
                            "--seed-results", str(out))
        assert code == 2
        err = capsys.readouterr().err
        assert "implementation" in err and "package_files" in err
    finally:
        sys.modules.pop("edited_probe", None)


def test_the_runtime_semantics_are_part_of_the_context(tmp_path, monkeypatch):
    """A seed scored before a change to how the runtime treats a crashed step (a crash
    retried, a spent one leaving no answer) is not reused (`node_retry.SEMANTICS`); nor is
    a run's evidence resumed across one (`profiles.resolve`)."""
    from refactory_dsh import node_retry, profiles

    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        monkeypatch.setattr(node_retry, "SEMANTICS", "an older runtime")
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        with pytest.raises(LoopError, match="runtime_semantics"):
            loop.init()
        before = profiles.resolve(tmp_path, MODE)
        monkeypatch.setattr(node_retry, "SEMANTICS", "pass-crash-retry/1")
        after = profiles.resolve(tmp_path, MODE)
        assert before["evaluation_context"]["runtime_semantics"] == "an older runtime"
        assert before["context_hash"] != after["context_hash"]


# ── the run's seed, and every setting that decides what an item scores ───────


def test_a_seed_42_cache_is_refused_for_a_seed_43_run(tmp_path):
    """A Task may shuffle an item by the run's seed (LiveMath's answer choices), so a
    seed-42 export would serve a seed-43 run scores of other questions: refused, naming
    the field."""
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path, seed=42)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed=43, seed_results=str(cache))
        with pytest.raises(LoopError, match=r"different score context \(seed\)"):
            loop.init()


def test_a_seed_42_cache_still_serves_runs_with_other_search_settings(tmp_path):
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path, seed=42)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed=42, seed_results=str(cache),
                     optimizer="random", strategy="tree_search", parent_picker="map_elites",
                     epochs=5, batch_size=3)
        loop.init()
        seed = loop.population[0]
        Run(_loop=loop).evaluate(seed)
        Run(_loop=loop).evaluate(seed, part="val")
        assert seed.raw_score == 0.25 and loop.rollouts == 0
        assert [h["part"] for h in _events(tmp_path / "reuse", "seed_cache_hit")] == ["train", "val"]


SEARCH_ONLY_CHANGES = {"strategy": "tree_search", "optimizer": "hill_climb", "val_gate": True,
                       "portfolio": "prompt+pipeline", "parent_picker": "map_elites", "epochs": 9}


@pytest.mark.parametrize("field", sorted(SEARCH_ONLY_CHANGES))
def test_search_only_settings_leave_the_score_context_alone(field):
    """#616's reuse across search settings: strategy, optimizer, val gate, portfolio,
    parent picker and epochs do not decide what an item scores."""
    from test_parts import PartsProbe as mode_class

    base = LoopConfig(seed=42)
    changed = LoopConfig(seed=42, **{field: SEARCH_ONLY_CHANGES[field]})
    assert seed_cache.score_context(changed, mode_class) == seed_cache.score_context(base, mode_class)


def test_every_loop_setting_is_in_the_score_context_or_left_out_for_a_reason():
    """A new LoopConfig field has to be placed: in the score context, or left out with
    the reason it cannot change what an item scores."""
    import dataclasses

    fields = {field.name for field in dataclasses.fields(LoopConfig)}
    scored, left_out = set(seed_cache.SCORED_FIELDS), set(seed_cache.SEARCH_ONLY_FIELDS)
    assert not scored & left_out
    assert fields == scored | left_out, (sorted(fields - scored - left_out),
                                         sorted((scored | left_out) - fields))
    assert all(seed_cache.SEARCH_ONLY_FIELDS[field].strip() for field in left_out)


SCORED_CHANGES = {"seed": 43, "provider_config": {"p": {"models": []}},
                  "node_routes": {"solve": {"model": "m"}}, "role_routes": {"solver": {"model": "m"}},
                  "evaluator": "dsh", "dsh_timeout_s": 60, "max_steps": 10, "repeats": 3,
                  "intent": "answer the questions", "omit_steering": True}


@pytest.mark.parametrize("field", seed_cache.SCORED_FIELDS)
def test_each_scored_setting_changes_the_score_context_naming_it(field):
    from test_parts import PartsProbe as mode_class

    base = seed_cache.score_context(LoopConfig(seed=42), mode_class)
    changed = seed_cache.score_context(
        LoopConfig(**{"seed": 42, field: SCORED_CHANGES[field]}), mode_class)
    assert seed_cache.context_diff(changed, base) == [field]


def test_another_rollout_seed_measures_the_part_and_says_why(tmp_path):
    """The cached part was measured with one rollout seed; an evaluation asking for
    another (a gated run's per-epoch train seed against an ungated run's) runs for real
    and is journaled as a miss, while a part asked for with the cached seed is served."""
    with _patch_mode(PartsProbe):
        cache = _source_run(tmp_path)
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(cache))
        loop.init()
        seed = loop.population[0]
        Run(_loop=loop).evaluate(seed, seed=1002)
        Run(_loop=loop).evaluate(seed, part="val")
        assert seed.raw_score == 1.0 and loop.rollouts == len(PARTS["train"])
        misses = _events(tmp_path / "reuse", "seed_cache_miss")
        assert [(m["part"], m["reason"]) for m in misses] == [("train", "rollout seed 1002, cached None")]
        assert [h["part"] for h in _events(tmp_path / "reuse", "seed_cache_hit")] == ["val"]


def test_the_export_records_each_parts_rollout_seed(tmp_path, exported):
    """Run A (seed 3, val gate) rolled its seed out on train with SkillOpt's epoch-1 seed
    and on val and test with the run's own."""
    cache = json.loads(exported.read_text())
    assert cache["score_context"]["seed"] == 3
    assert {part: entry["seed"] for part, entry in cache["parts"].items()} == {
        "train": 1004, "val": 3, "test": 3}


def test_an_ungated_run_measures_train_rather_than_reuse_a_gated_export(tmp_path, exported):
    """Without the gate (a run with no val items) train rollouts use the run's seed, not
    the gated export's per-epoch one, so train is measured for real and the miss says
    why."""
    no_val = _manifest(tmp_path, {key: ids for key, ids in PARTS.items() if key != "val"},
                       name="noval.json")
    with _patch_mode(PartsProbe):
        code, root = _main(tmp_path, "B", "--optimizer", "hill_climb", "--epochs", "1",
                           "--repeats", "3", "--seed-results", str(exported), manifest=no_val)
    assert code == 0
    misses = _events(root, "seed_cache_miss")
    assert [(m["part"], m["reason"]) for m in misses] == [("train", "rollout seed 3, cached 1004")]
    assert "train" not in {h["part"] for h in _events(root, "seed_cache_hit")}


def test_a_cached_part_with_items_that_died_is_measured(tmp_path):
    """A cached train part one of whose items died with its run's processes (its model
    calls kept failing on the wire) is no result to replay: train is measured here and
    the miss says why; val is still served."""
    with _patch_mode(PartsProbe):
        path = _source_run(tmp_path)
        cache = json.loads(path.read_text())
        entry = cache["parts"]["train"]
        records = (entry.get("trials") or [entry["result"]])[0]["item_records"]
        records[0].update(passed=False, score=0.0, error_kind="RuntimeError",
                          error_text="work: CompletionFailed: model completion failed on the "
                                     "wire 3 time(s); last cause: dsh: TRANSPORT: fetch failed")
        path.write_text(json.dumps(cache))
        loop = _loop(tmp_path / "reuse", _manifest(tmp_path), seed_results=str(path))
        loop.init()
        seed = loop.population[0]
        Run(_loop=loop).evaluate(seed)
        Run(_loop=loop).evaluate(seed, part="val")
        assert seed.raw_score == 1.0 and loop.rollouts == len(PARTS["train"])
        misses = _events(tmp_path / "reuse", "seed_cache_miss")
        assert [(m["part"], m["reason"]) for m in misses] == [
            ("train", "cached items died with their run's processes")]
        assert [h["part"] for h in _events(tmp_path / "reuse", "seed_cache_hit")] == ["val"]


def test_the_export_reads_the_occasion_its_result_names(tmp_path):
    """An interrupted val rollout took occasion 0 and left no result; the first result is
    occasion 1, and the export reads that occasion's records, not occasion 0's."""
    with _patch_mode(PartsProbe):
        loop = _loop(tmp_path / "source", _manifest(tmp_path))
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        run.evaluate(seed)
        stale = tmp_path / "source" / "runs" / seed.id / "occasions" / "val_0"
        stale.mkdir(parents=True)
        (stale / "item_records.json").write_text(json.dumps([{"id": "stale", "score": 0.0}]))
        loop._occasion_counters.setdefault(seed.id, {})["val"] = 1
        run.evaluate(seed, part="val")
        cache = seed_cache.export(tmp_path / "source", project=Path("."))
    assert cache["parts"]["val"]["workdir"].endswith("occasions/val_1")
    assert cache["parts"]["val"]["items"] == PARTS["val"]
