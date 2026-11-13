"""Optimization profiles end to end: every setting a default-loop experiment uses
round-trips into ``resolved-profile.json``, a run launched from a profile behaves as
the same settings given as flags, overrides are applied and recorded, versions stay
immutable, and an unknown or mistyped setting is refused.

Every run here uses the walker over deterministic nodes and the random or hill-climb
optimizer, so no model is called; the autoresearch profile is only resolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import loop as loop_module, nn, profiles, strategies
from refactory_dsh.loop import Loop

from test_parts import MODE, TASK_REF, PartsProbe, _patch_mode

PARTS = {"train": ["alpha", "beta", "gamma", "delta"], "val": ["epsilon"], "test": ["zeta"]}


class ProfileProbe(PartsProbe):
    """PartsProbe with a searchable knob that moves the score and LiveMath's frozen
    ``data_parallelism`` resource knob on its data node."""

    name = "profile_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {
            "bias": nn.Knob(0.5, "float", lo=0.0, hi=1.0, description="distance from the best score"),
            "data_parallelism": nn.Knob(1, "int", lo=1, hi=8, frozen=True,
                                        description="concurrent data-node items"),
        }

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node(
            "data", "data", subgraph_entry="work", subgraph_exit="work", task_ref=TASK_REF,
            writes=("current_item.json",), produces=("item_scores",),
            parallelism_knob="data_parallelism", knobs=("data_parallelism",),
        ))
        graph.add(nn.Node(
            "work", "op", call="loop_ops:record_item", knobs=("bias",),
            reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
        ))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        return super().score(run) * (1.0 - abs(float(self.values()["bias"]) - 0.3))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project and Refactory home, so nothing touches the real ones."""
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("REFACTORY_WORKER_CONTEXT", raising=False)
    (tmp_path / "parts.json").write_text(json.dumps(PARTS))
    with _patch_mode(ProfileProbe):
        yield tmp_path


def _experiment(project: Path) -> dict[str, Any]:
    """The default-loop settings the LiveMath experiment runs with, scaled down."""
    return {
        "optimizer": "random", "strategy": "default", "train_minibatch": 2,
        "repeats": 3, "holdout": 0.0, "parts_manifest": str(project / "parts.json"),
        "parsimony": 0.02, "knobs": {"data_parallelism": "3"}, "epochs": 2, "batch_size": 2,
        "edit_preset": "conservative", "allowed_ops": ["set_knob"], "allow_code_nodes": False,
        "max_evaluations": 12, "max_elapsed_s": 600.0, "seed": 5,
        "provider": "fake-provider", "model": "task-model",
        "role_routes": {"optimizer": {"model": "proposer-model"}},
        "node_routes": {"work": {"model": "task-model"}},
    }


#: The experiment's settings that ``refactory optimize`` takes as flags, as flags.
def _experiment_flags() -> list[str]:
    return [
        "--optimizer", "random", "--strategy", "default", "--knob", "data_parallelism=3",
        "--epochs", "2", "--batch-size", "2", "--preset", "conservative",
        "--max-evaluations", "12", "--seed", "5", "--provider", "fake-provider", "--model", "task-model",
    ]


#: The experiment's settings the flags above give.
FLAGGED = ("optimizer", "strategy", "knobs", "epochs", "batch_size", "edit_preset",
           "max_evaluations", "seed", "provider", "model")


def _run(project: Path, name: str, *argv: str) -> tuple[int, Path]:
    root = project / name
    code = loop_module.main([MODE, "--project", str(project), "--runs-root", str(root), *argv])
    return code, root


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_text())


def _board(root: Path) -> list[tuple]:
    return [(row["id"], row["score"], row["raw_score"]) for row in _read(root, "report.json")["leaderboard"]]


def test_every_experiment_setting_round_trips_into_the_snapshot_and_the_run(project):
    settings = _experiment(project)
    profiles.save(project, "experiment", {"settings": settings, "contract": {"objective": "LiveMath"}})
    code, root = _run(project, "run", "--profile", "experiment", "--profile-version", "1")
    assert code == 0
    snapshot, config = _read(root, profiles.SNAPSHOT), _read(root, "config.json")
    assert snapshot["profile"] == {"name": "experiment", "version": 1}
    for key, value in settings.items():
        assert snapshot["settings"][key] == value, key
        assert config[key] == value, key
    # The manifest has val items, so the run decided on the validation gate.
    assert config["val_gate"] is True and "val_gate" not in snapshot["settings"]
    # The run did what the profile says: the manifest's parts, three trials per
    # measurement, the starting knob, and the gate's val occasions.
    assert _read(root, "parts.json") == PARTS
    journal = [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]
    seed = next(r["state"] for r in journal if r["event"] == "original_seed")
    assert seed["knobs"]["data_parallelism"]["value"] == 3
    scores = [r for r in journal if r["event"] == "score"]
    assert scores and all(r["measurements"]["count"] == 3 for r in scores)
    minibatches = [r["ids"] for r in journal if r["event"] == "train_minibatch"]
    assert minibatches and all(len(ids) == 2 for ids in minibatches)
    assert any(r["event"] == "part_result" and r["part"] == "val" for r in journal)
    assert all(e["op"] == "set_knob" for c in journal if c["event"] == "candidate"
               for e in c.get("edits") or [])


def test_a_profile_run_matches_the_same_settings_given_as_flags(project):
    """The settings ``refactory optimize`` takes as flags, given as flags over a profile
    holding the rest, run as the profile holding all of them does."""
    profiles.save(project, "experiment", {"settings": _experiment(project)})
    profiles.save(project, "rest", {"settings": {
        key: value for key, value in _experiment(project).items() if key not in FLAGGED}})
    code_a, by_profile = _run(project, "profile", "--profile", "experiment")
    code_b, by_flags = _run(project, "flags", "--profile", "rest", *_experiment_flags())
    assert code_a == code_b == 0
    a, b = _read(by_profile, profiles.SNAPSHOT), _read(by_flags, profiles.SNAPSHOT)
    assert a["settings"] == b["settings"]
    assert a["context_hash"] == b["context_hash"]
    assert _board(by_profile) == _board(by_flags)
    assert len(_board(by_profile)) > 1  # the search moved the knob, so rows differ


def test_cli_overrides_on_a_profile_apply_and_are_recorded(project):
    profiles.save(project, "experiment", {"settings": _experiment(project)})
    code, root = _run(project, "run", "--profile", "experiment", "--epochs", "1",
                      "--seed", "9", "--knob", "bias=0.4")
    assert code == 0
    snapshot = _read(root, profiles.SNAPSHOT)
    assert snapshot["settings"]["epochs"] == 1
    assert snapshot["settings"]["seed"] == 9
    # --knob adds to the profile's knobs; every setting not given keeps its value.
    assert snapshot["settings"]["knobs"] == {"data_parallelism": "3", "bias": "0.4"}
    assert snapshot["settings"]["parts_manifest"] == str(project / "parts.json")
    assert _read(root, "config.json")["val_gate"] is True
    assert snapshot["overrides"] == {"epochs": 1, "seed": 9, "knob": ["bias=0.4"]}
    assert _read(root, "report.json")["epochs_completed"] == 1


def test_a_saved_profile_with_removed_settings_still_loads_and_runs(project):
    """A profile saved before ``val_gate``, ``fresh_evidence`` and
    ``max_edits_per_proposal`` were removed still loads: the keys are dropped (its
    file is left as it was), ``val_gate: false`` (the default before #627) reads as
    unset, and the run decides the gate from its parts manifest."""
    old = {**_experiment(project), "val_gate": False, "fresh_evidence": False,
           "max_edits_per_proposal": 2}
    profiles.save(project, "old", {"settings": _experiment(project)})
    path = project / profiles.PROFILE_ROOT / "old" / "1.json"
    saved = json.loads(path.read_text())
    saved["settings"].update({key: old[key] for key in profiles.OBSOLETE_SETTINGS})
    path.write_text(json.dumps(saved))
    loaded = profiles.load(project, "old")["settings"]
    assert not profiles.OBSOLETE_SETTINGS & set(loaded)
    assert json.loads(path.read_text())["settings"]["val_gate"] is False, "file untouched"
    code, root = _run(project, "run", "--profile", "old")
    assert code == 0
    assert _read(root, "config.json")["val_gate"] is True
    assert not profiles.OBSOLETE_SETTINGS & set(_read(root, profiles.SNAPSHOT)["settings"])


def test_a_profile_alone_supplies_the_parts_manifest(project):
    """A profile's parts manifest reaches the run without --parts (it was dropped,
    and the run refused the validation gate for want of a val part)."""
    settings = {k: v for k, v in _experiment(project).items() if k in (
        "optimizer", "parts_manifest", "holdout", "epochs", "batch_size", "edit_preset")}
    profiles.save(project, "parts", {"settings": settings})
    code, root = _run(project, "run", "--profile", "parts")
    assert code == 0
    assert _read(root, "config.json")["parts_manifest"] == str(project / "parts.json")
    assert _read(root, "parts.json") == PARTS


class OptionProbe(Loop):
    """A library-style strategy that reads one option, to see options arrive."""

    name = "option_probe"
    accepted_options = ("depth", "width")
    seen: list[dict[str, Any]] = []

    def configure(self, *, model: Any = None, **options: Any) -> None:
        super().configure(model=model, **options)
        OptionProbe.seen.append(dict(options))


@pytest.fixture
def option_probe():
    OptionProbe.seen = []
    strategies.register(OptionProbe.name, OptionProbe)
    yield OptionProbe
    strategies.STRATEGIES.pop(OptionProbe.name, None)


def test_profile_strategy_options_reach_the_strategy_and_flags_add_to_them(project, option_probe):
    base = {"optimizer": "random", "strategy": "option_probe", "epochs": 1, "batch_size": 1,
            "edit_preset": "knob_only", "strategy_options": {"depth": 2, "width": 1}}
    profiles.save(project, "options", {"settings": base})
    code, root = _run(project, "run", "--profile", "options")
    assert code == 0
    assert option_probe.seen[-1] == {"depth": 2, "width": 1}
    assert _read(root, profiles.SNAPSHOT)["settings"]["strategy_options"] == {"depth": 2, "width": 1}
    code, root = _run(project, "again", "--profile", "options", "--strategy-option", "depth=3")
    assert code == 0
    assert option_probe.seen[-1] == {"depth": 3, "width": 1}
    assert _read(root, "config.json")["strategy_options"] == {"depth": 3, "width": 1}


def test_skillopt_refuses_an_option_it_does_not_read():
    strategies.check_options("skillopt", {"max_epochs": 1, "knobs": "skill", "early_stop_patience": 3})
    with pytest.raises(ValueError, match="no option max_epoch"):
        strategies.check_options("skillopt", {"max_epoch": 1})


@pytest.mark.parametrize("spec, message", [
    ({"settings": {"epoch": 1}}, "unknown search settings: epoch"),
    ({"setting": {"epochs": 1}}, "unknown profile fields: setting"),
    ({"settings": {"repeats": True}}, "repeats must be an integer"),
    ({"settings": {"seed": "abc"}}, "seed must be an integer"),
    ({"settings": {"omit_steering": "yes"}}, "omit_steering must be true or false"),
    ({"settings": {"parts_manifest": 3}}, "parts_manifest must be a string or null"),
    ({"settings": {"strategy_options": {"bogus": 1}}}, "default strategy has no option bogus"),
    ({"settings": {"strategy_options": "depth=2"}}, "strategy_options must be an object"),
    ({"settings": {"allowed_ops": ["no_such_op"]}}, "allowed_ops must belong"),
    ({"settings": {"optimizer": "autoreserch"}}, "unknown optimizer"),
])
def test_an_unknown_or_mistyped_setting_is_refused_and_saves_nothing(project, spec, message):
    with pytest.raises(profiles.ProfileError, match=message):
        profiles.save(project, "bad", spec)
    assert not (project / profiles.PROFILE_ROOT / "bad").exists()


def test_a_switch_given_as_1_is_checked_but_kept_as_given():
    """Rewriting 1 to true would change the settings digest of a run started with
    `--omit-steering 1`, so it could not resume."""
    settings = profiles.normalize({"omit_steering": 1})
    assert settings["omit_steering"] == 1 and settings["omit_steering"] is not True
    with pytest.raises(profiles.ProfileError, match="omit_steering must be true or false"):
        profiles.normalize({"omit_steering": 2})


def test_a_run_written_in_the_old_form_resumes_under_the_same_context_hash(project):
    """A run started before this change (a switch stored as 1, a config without
    strategy_options) resumes: its context hash is unchanged."""
    # A switch saved as 1, as the command line once passed it.
    profiles.save(project, "old-form", {"settings": {"omit_steering": 1}})
    code, root = _run(project, "old", "--optimizer", "random", "--epochs", "1", "--batch-size", "1",
                      "--preset", "knob_only", "--profile", "old-form")
    assert code == 0
    snapshot = _read(root, profiles.SNAPSHOT)
    before = snapshot["context_hash"]
    # Stored as given, as a run started before this change stored it.
    assert snapshot["settings"]["omit_steering"] == 1 and snapshot["settings"]["omit_steering"] is not True
    saved = _read(root, "config.json")
    assert saved["omit_steering"] == 1 and saved["omit_steering"] is not True
    saved.pop("strategy_options")
    (root / "config.json").write_text(json.dumps(saved))
    code = loop_module.main([MODE, "--project", str(project), "--runs-root", str(root),
                             "--resume", "--epochs", "2"])
    assert code == 0
    journal = [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]
    resumed = [r["resolved_profile"] for r in journal if r["event"] == "resume_configuration"]
    assert resumed and resumed[-1]["context_hash"] == before
    # The flag given on resume joins the overrides the launch recorded over its profile.
    assert resumed[-1]["overrides"] == {"optimizer": "random", "epochs": 2, "batch_size": 1,
                                        "edit_preset": "knob_only"}


def test_another_strategy_drops_the_profiles_strategy_options(project, option_probe):
    profiles.save(project, "options", {"settings": {
        "optimizer": "random", "strategy": "option_probe", "epochs": 1, "batch_size": 1,
        "edit_preset": "knob_only", "strategy_options": {"depth": 2}}})
    code, root = _run(project, "run", "--profile", "options", "--strategy", "default")
    assert code == 0
    snapshot = _read(root, profiles.SNAPSHOT)
    assert snapshot["settings"]["strategy"] == "default"
    assert snapshot["settings"]["strategy_options"] == {}
    assert snapshot["overrides"] == {"strategy": "default", "strategy_options": {}}


def test_a_strategy_that_declares_no_options_is_not_checked(option_probe):
    """Only a strategy's own accepted_options counts: a Loop subclass that declares
    none is unchecked, not refused everything its base refuses."""

    class Undeclared(Loop):
        name = "undeclared_probe"

    strategies.register(Undeclared.name, Undeclared)
    try:
        strategies.check_options("undeclared_probe", {"anything": 1})
        with pytest.raises(ValueError, match="no option bogus"):
            strategies.check_options("default", {"bogus": 1})
        with pytest.raises(ValueError, match="no option bogus"):
            strategies.check_options("option_probe", {"bogus": 1})
    finally:
        strategies.STRATEGIES.pop(Undeclared.name, None)


def test_a_misspelled_flag_value_or_strategy_option_is_refused_at_launch(project, capsys):
    assert _run(project, "a", "--optimizer", "random", "--strategy-option", "bogus=1")[0] == 2
    assert "no option bogus" in capsys.readouterr().err
    profiles.save(project, "p", {"settings": {"optimizer": "random", "knobs": {"no_such_knob": 1}}})
    assert _run(project, "b", "--profile", "p")[0] == 2
    assert "no knob 'no_such_knob'" in capsys.readouterr().err


def test_versions_are_immutable(project):
    first = profiles.save(project, "p", {"settings": {"optimizer": "random", "epochs": 1}})
    path = project / profiles.PROFILE_ROOT / "p" / "1.json"
    before = path.read_bytes()
    second = profiles.save(project, "p", {"settings": {"optimizer": "random", "epochs": 3}})
    assert (first["version"], second["version"]) == (1, 2)
    assert path.read_bytes() == before
    assert profiles.load(project, "p", 1)["settings"]["epochs"] == 1
    assert profiles.load(project, "p")["settings"]["epochs"] == 3
    code, root = _run(project, "run", "--profile", "p", "--profile-version", "1")
    assert code == 0
    snapshot = _read(root, profiles.SNAPSHOT)
    assert snapshot["profile"] == {"name": "p", "version": 1}
    assert snapshot["settings"]["epochs"] == 1
    with pytest.raises(profiles.ProfileError, match="version 9 does not exist"):
        profiles.load(project, "p", 9)


def test_a_save_never_overwrites_a_version_another_save_took(project, monkeypatch):
    profiles.save(project, "p", {"settings": {"optimizer": "random"}})
    root = project / profiles.PROFILE_ROOT / "p"
    real_glob = Path.glob
    raced = []

    def stale_glob(self, pattern):
        # The first listing misses version 1, as a concurrent save would see it.
        if self == root and not raced:
            raced.append(True)
            return iter(())
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", stale_glob)
    saved = profiles.save(project, "p", {"settings": {"optimizer": "random", "epochs": 5}})
    assert saved["version"] == 2
    assert profiles.load(project, "p", 1)["settings"]["epochs"] != 5


def test_max_evaluations_from_a_profile_bounds_the_run(project):
    profiles.save(project, "small", {"settings": {
        "optimizer": "random", "epochs": 6, "batch_size": 2, "edit_preset": "knob_only",
        "max_evaluations": 3, "max_elapsed_s": 300.0}})
    code, root = _run(project, "run", "--profile", "small")
    assert code == 0
    report = _read(root, "report.json")
    assert report["evaluations"] <= 3
    assert report["bound"] == "evaluations"
    assert _read(root, profiles.SNAPSHOT)["contract"]["termination"]["max_elapsed_s"] == 300.0


def test_an_autoresearch_profile_resolves_with_its_proposer_and_task_routes(project):
    """Resolved and handed to the model client, never run: the client stub stops the
    launch before any model is called."""
    settings = {**_experiment(project), "optimizer": "autoresearch"}
    profiles.save(project, "auto", {"settings": settings})
    snapshot = profiles.resolve(project, MODE, name="auto")
    assert snapshot["settings"]["optimizer"] == "autoresearch"
    assert snapshot["settings"]["role_routes"] == {"optimizer": {"model": "proposer-model"}}
    assert snapshot["settings"]["node_routes"] == {"work": {"model": "task-model"}}
    routing = snapshot["evaluation_context"]["routing"]
    assert routing["route"]["provider"] == "fake-provider"
    assert routing["route"]["model"] == "task-model"

    seen: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> Any:
        seen.append(kwargs)
        raise ValueError("stop before any model call")

    code = loop_module.main([MODE, "--project", str(project), "--runs-root", str(project / "r"),
                             "--profile", "auto"], _client_factory=factory)
    assert code == 2
    assert seen[0]["role_routes"] == {"optimizer": {"model": "proposer-model"}}
    assert seen[0]["node_routes"] == {"work": {"model": "task-model"}}
    assert (seen[0]["provider"], seen[0]["model"]) == ("fake-provider", "task-model")
    assert not (project / "r" / "journal.jsonl").exists()


def test_a_profile_with_a_parent_picker_and_a_portfolio_round_trips(project):
    """``parent_picker`` and ``portfolio`` pass the type check, survive save, load and
    resolve, change the context hash only when set, and reach the run."""
    from refactory_dsh.loop import LoopConfig

    auto = {**_experiment(project), "optimizer": "autoresearch",
            "train_minibatch": 0, "parent_picker": "map_elites", "portfolio": "prompt+pipeline"}
    profiles.save(project, "combo", {"settings": auto})
    assert profiles.load(project, "combo")["settings"]["parent_picker"] == "map_elites"
    snapshot = profiles.resolve(project, MODE, name="combo")
    assert snapshot["settings"]["parent_picker"] == "map_elites"
    assert snapshot["settings"]["portfolio"] == "prompt+pipeline"
    config = LoopConfig.from_dict(snapshot["settings"])
    assert (config.parent_picker, config.portfolio) == ("map_elites", "prompt+pipeline")
    plain = profiles.resolve(project, MODE, overrides={**auto, "parent_picker": "", "portfolio": ""})
    assert plain["evaluation_context"]["settings"] != snapshot["evaluation_context"]["settings"]
    for bad, message in (({"parent_picker": 3}, "parent_picker"),
                         ({"parent_picker": "tournament"}, "parent_picker"),
                         ({"portfolio": "everything"}, "portfolio"),
                         ({"optimizer": "random", "portfolio": "prompt+pipeline"}, "portfolio")):
        with pytest.raises(profiles.ProfileError, match=message):
            profiles.normalize({**auto, **bad})

    # A random proposer with a picker runs from the profile, no model needed.
    profiles.save(project, "picked", {"settings": {**_experiment(project),
                                                   "train_minibatch": 0,
                                                   "parent_picker": "map_elites"}})
    code, root = _run(project, "picked-run", "--profile", "picked")
    assert code == 0
    assert _read(root, "config.json")["parent_picker"] == "map_elites"
    assert _read(root, "report.json")["optimizer"] == "random+map_elites"


# ── the short command line ──────────────────────────────────────────────────


def _parse(argv: list[str], *, cli: bool) -> Any:
    """Parse ``refactory optimize`` (``cli``) or ``python -m refactory_dsh.loop``."""
    import argparse

    if cli:
        from refactory_dsh.cli import build_parser
        return build_parser().parse_args(["optimize", MODE, *argv])
    parser = argparse.ArgumentParser(prog="python -m refactory_dsh.loop")
    parser.add_argument("mode")
    profiles.add_cli_arguments(parser)
    return parser.parse_args([MODE, *argv])


@pytest.mark.parametrize("cli", [True, False], ids=["refactory-optimize", "loop-main"])
def test_every_other_setting_is_refused_naming_its_profile_key(cli, capsys):
    """Every ``LoopConfig`` setting is a flag of the short list or refused with the
    profile key that sets it and a ``profile-save`` spec, never "unrecognized"."""
    flagged = set(profiles.CLI_SETTINGS)
    for key in profiles.defaults():
        if key in flagged:
            continue
        flag = "--" + key.replace("_", "-")
        with pytest.raises(SystemExit) as refused:
            _parse([flag, "1"], cli=cli)
        assert refused.value.code == 2
        err = capsys.readouterr().err
        assert f"{flag} is not a command-line flag; it is the profile setting {key!r}" in err, err
        assert "refactory profile-save --name NAME --spec" in err and "usage:" not in err
    with pytest.raises(SystemExit):
        _parse(["--parts", "p.json"], cli=cli)
    err = capsys.readouterr().err
    assert "the profile setting 'parts_manifest'" in err
    assert '{"settings": {"parts_manifest": "p.json"}}' in err
    with pytest.raises(SystemExit):
        _parse(["--holdout", "0"], cli=cli)
    assert '{"settings": {"holdout": 0}}' in capsys.readouterr().err


def test_the_short_list_parses_and_reaches_the_overrides():
    """The flags the short list keeps reach ``cli_overrides`` as their settings."""
    args = _parse(["--seed", "4", "--epochs", "3", "--batch-size", "2", "--max-evaluations", "9",
                   "--optimizer", "random", "--preset", "knob_only", "--evaluator", "walker",
                   "--strategy", "default", "--intent", "why", "--seed-results", "s.json",
                   "--max-items-in-flight", "6", "--provider", "p", "--model", "m",
                   "--effort", "low", "--knob", "a=1", "--strategy-option", "b=2"], cli=False)
    assert profiles.cli_overrides(args) == {
        "seed": 4, "epochs": 3, "batch_size": 2, "max_evaluations": 9, "optimizer": "random",
        "edit_preset": "knob_only", "evaluator": "walker", "strategy": "default",
        "intent": "why", "seed_results": "s.json", "max_items_in_flight": 6, "provider": "p",
        "model": "m", "effort": "low", "knob": ["a=1"], "strategy_option": ["b=2"]}


def test_refactory_optimize_forwards_its_flags_to_the_loop(monkeypatch):
    """``refactory optimize`` hands the loop exactly the flags it was given."""
    from refactory_dsh import cli

    forwarded: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: forwarded.append([module, *argv]) or 0)
    assert cli.main(["optimize", MODE, "--epochs", "2", "--preset", "knob_only", "--resume",
                     "--project", "/p", "--knob", "a=1", "--knob", "b=2"]) == 0
    module, *argv = forwarded[0]
    assert module == "refactory_dsh.loop"
    assert argv[0] == MODE
    pairs = [argv[i:i + 2] for i in range(1, len(argv)) if argv[i].startswith("--")]
    assert ["--epochs", "2"] in pairs and ["--preset", "knob_only"] in pairs
    assert ["--project", "/p"] in pairs and "--resume" in argv
    assert argv.count("--knob") == 2


def test_a_resumed_run_recorded_with_removed_settings_says_its_behaviour_changes(project):
    """A run recorded with ``fresh_evidence: false`` and ``max_edits_per_proposal: 2``
    (both removed in #640) resumes, and its log says each now differs; one recorded at
    today's values says nothing."""
    code, root = _run(project, "old", "--optimizer", "random", "--epochs", "1",
                      "--batch-size", "1", "--preset", "knob_only")
    assert code == 0
    saved = _read(root, "config.json")
    (root / "config.json").write_text(json.dumps(
        {**saved, "fresh_evidence": False, "max_edits_per_proposal": 2}))
    code = loop_module.main([MODE, "--project", str(project), "--runs-root", str(root),
                             "--resume", "--epochs", "2"])
    assert code == 0
    log = (root / "strategy.log").read_text()
    assert "recorded with fresh_evidence false" in log
    assert "recorded with max_edits_per_proposal 2" in log and "the limit is 4" in log
    assert "fresh_evidence" not in _read(root, "config.json"), "the record is rewritten"

    code, same = _run(project, "same", "--optimizer", "random", "--epochs", "1",
                      "--batch-size", "1", "--preset", "knob_only")
    saved = _read(same, "config.json")
    (same / "config.json").write_text(json.dumps(
        {**saved, "fresh_evidence": True, "max_edits_per_proposal": 4}))
    code = loop_module.main([MODE, "--project", str(project), "--runs-root", str(same),
                             "--resume", "--epochs", "2"])
    assert code == 0
    log = (same / "strategy.log").read_text() if (same / "strategy.log").exists() else ""
    assert "fresh_evidence" not in log and "max_edits_per_proposal" not in log
