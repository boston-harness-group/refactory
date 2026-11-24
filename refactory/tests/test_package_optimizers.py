"""Parent pickers and proposers as package building blocks (``Package.optimizers``).

A package lists optimizers the way it lists strategies, and a run of the project the
package is in selects them by name through the normal settings and the command line
(``optimizer``, ``parent_picker``, ``strategy``). Two routes, each one copy: the
library's own table through ``optim.register``, and the enabled packages through
``registry.contributed``; a name clash between them is refused. A picker that asks
(``uses_item_scores``) is handed each judged candidate's per-item validation scores
(``ProposalContext.val_item_scores``), keyed by the run's HMAC of the item id, never a
raw id, never test, and never shown to the proposer. Each test fails when the rule it
names is reverted.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

import gepa_ops
from refactory_dsh import optim, profiles, registry, strategies
from refactory_dsh.loop import Loop, LoopConfig, default_selection, opaque_item_key
from refactory_dsh.nn import Package
from refactory_dsh.nn.package import PackageError
from refactory_dsh.outerloop import OuterLoopError, Run

from test_gepa import MODE_ID, OneKnob, _patch_mode

PACKAGE = "probe_blocks"

#: A package contributing one parent picker and one proposer. The picker records what
#: it is handed; the proposer writes the next of a fixed list of instructions.
SOURCE = textwrap.dedent('''
    from refactory_dsh.nn import Package
    from refactory_dsh.optim import Optimizer, ParentSelector, Proposal
    from refactory_dsh.outerloop import OuterLoop

    SEEN = []
    PROPOSED = []
    TEXTS = ["Always answer ALPHA.", "Always answer BETA.", "Say ALPHA.", "Say BETA."]


    class ProbePicker(ParentSelector):
        """Records the per-item val scores it is handed; picks the best by val."""

        name = "probe_picker"
        policy = "the probe's pick"
        uses_item_scores = True

        def select(self, ctx, n, parents):
            SEEN.append({"val_item_scores": {k: dict(v) for k, v in ctx.val_item_scores.items()},
                         "parents": [p.id for p in parents]})
            return self._ranked(ctx, parents)[:n]


    class ProbeProposer(Optimizer):
        """Sets the instruction knob to the next text."""

        name = "probe_proposer"

        def __init__(self, *, k=3, seed=0):
            super().__init__(k=k)

        def propose(self, ctx, n, parents):
            text = TEXTS[len(PROPOSED) % len(TEXTS)]
            PROPOSED.append({"parents": [p.id for p in parents], "text": text,
                             "item_scores": dict(ctx.val_item_scores)})
            return [Proposal(edit={"op": "set_knob", "name": "instruction", "value": text},
                             parent=parents[0].id)]


    class ProbeLoop(OuterLoop):
        """An outer loop the package contributes, for name resolution."""

        name = "probe_loop"
        accepted_options = ()


    PACKAGE = Package(name="probe_blocks", description="probe optimizers", version="1.0.0",
                      optimizers=(ProbePicker, ProbeProposer), strategies=(ProbeLoop,))
''')


def _project(tmp_path: Path, source: str = SOURCE, name: str = PACKAGE) -> Path:
    project = tmp_path / "project"
    package = registry.workspace_root(project) / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source)
    return project


def _state(loop: Loop) -> dict[str, Any]:
    """What the run's own picker and proposer recorded. Read from each class's module
    globals: discovery reloads a workspace package, so the module now in
    ``sys.modules`` may be a later copy, and the two may come from different loads."""
    return {"SEEN": type(loop.optimizer.picker).select.__globals__["SEEN"],
            "PROPOSED": type(loop.optimizer.proposer).propose.__globals__["PROPOSED"]}


def _run(tmp_path: Path, project: Path, *, val: bool = True, **config: Any) -> Loop:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parts = tmp_path / "parts.json"
    manifest = {"train": gepa_ops.TRAIN_IDS, "test": gepa_ops.TEST_IDS}
    if val:
        manifest["val"] = gepa_ops.VAL_IDS
    parts.write_text(json.dumps(manifest))
    cfg = LoopConfig(**{
        "epochs": 3, "batch_size": 1, "optimizer": "probe_proposer",
        "parent_picker": "probe_picker", "seed": 1, "runs_root": tmp_path / "run",
        "allowed_ops": ("set_knob",), "parts_manifest": str(parts), "val_gate": val,
        "selection": "pool" if val else "", "max_evaluations": 50, "holdout": 0.0,
        **config,
    })
    with _patch_mode(OneKnob):
        loop = strategies.create("default", MODE_ID, cfg, project=project)
        loop.run()
    return loop


# ── the package route, end to end ──────────────────────────────────────────


def test_a_package_picker_and_proposer_drive_a_run_through_the_normal_settings(tmp_path):
    project = _project(tmp_path)
    assert {"probe_picker", "probe_proposer"} <= set(optim.available(project))
    assert "probe_picker" in optim.picker_names(project)
    assert "probe_proposer" not in optim.picker_names(project)
    assert "probe_picker" not in optim.REGISTRY, "a package optimizer joined core's table"
    loop = _run(tmp_path, project)
    state = _state(loop)
    assert loop.optimizer.name == "probe_proposer+probe_picker"
    assert type(loop.optimizer.picker).__module__ == f"refactory_packages.{PACKAGE}"
    assert state["PROPOSED"], "the package proposer wrote no proposal"
    assert state["SEEN"], "the package picker was never asked"
    picked = {p for call in state["SEEN"] for p in call["parents"]}
    assert all(set(p["parents"]) <= picked for p in state["PROPOSED"])
    # Every proposal went to the picker's choice.
    texts = {c.state["knobs"]["instruction"]["value"] for c in loop.population}
    assert "Always answer ALPHA." in texts


def test_a_picker_gets_per_item_val_scores_keyed_without_raw_ids(tmp_path):
    project = _project(tmp_path)
    loop = _run(tmp_path, project)
    seen = [s for s in _state(loop)["SEEN"] if s["val_item_scores"]]
    assert seen, "no picker call carried per-item val scores"
    secret = loop.item_key_secret
    val_keys = {opaque_item_key(i, secret): i for i in gepa_ops.VAL_IDS}
    raw = set(gepa_ops.TRAIN_IDS) | set(gepa_ops.VAL_IDS) | set(gepa_ops.TEST_IDS)
    test_keys = {opaque_item_key(i, secret) for i in gepa_ops.TEST_IDS}
    by_id = {c.id: c for c in loop.population}
    for call in seen:
        for cid, scores in call["val_item_scores"].items():
            # Val only: exactly the val items' keys, never a raw id, never a test key.
            assert set(scores) == set(val_keys)
            assert not set(scores) & raw
            assert not set(scores) & test_keys
            # Keyed as Run.part_item_scores keys them, and the scores are the items'.
            assert scores == loop.part_item_scores(cid, "val")
            text = by_id[cid].state["knobs"]["instruction"]["value"]
            for key, score in scores.items():
                passed, _why = gepa_ops.judge(val_keys[key], {"instruction": text})
                assert score == (1.0 if passed else 0.0)
    # The specialists win different items: the scores are per item, not a mean.
    last = seen[-1]["val_item_scores"]
    assert len({tuple(sorted(s.items())) for s in last.values()}) > 1


def test_without_the_gate_a_picker_gets_no_item_scores(tmp_path):
    project = _project(tmp_path)
    loop = _run(tmp_path / "no_val", project, val=False)
    seen = _state(loop)["SEEN"]
    assert seen and all(not s["val_item_scores"] for s in seen)
    # A run whose gate judged candidates hands them over only under the default
    # loop's gate: not when the gate is off, nor to a custom strategy's picker
    # (which keeps its own gate), as with ``val_scores``.
    gated = _run(tmp_path / "val", project)
    assert gated._val_item_scores() and gated._val_scores()
    gated.config.val_gate = False
    assert gated._val_item_scores() == {} == gated._val_scores()
    gated.config.val_gate = True
    gated._custom_strategy = object()
    assert gated._val_item_scores() == {} == gated._val_scores()


def test_a_proposer_never_sees_item_scores(tmp_path):
    """The per-item val scores are the picker's: the proposer paired with it is handed
    a context without them, though the picker had them that same step."""
    project = _project(tmp_path)
    loop = _run(tmp_path, project)
    state = _state(loop)
    assert any(s["val_item_scores"] for s in state["SEEN"])
    assert state["PROPOSED"] and all(p["item_scores"] == {} for p in state["PROPOSED"])


def test_only_a_picker_that_asks_gets_item_scores(tmp_path, monkeypatch):
    """A picker that does not compare items (``uses_item_scores`` False, every core
    picker but pareto) is handed none, and the loop does not work them out."""
    project = _project(tmp_path, SOURCE.replace("uses_item_scores = True\n", ""))
    from refactory_dsh import loop as loop_module

    calls: list[str] = []
    real = loop_module.Loop.part_item_scores

    def counted(self, candidate_id, part="val"):  # type: ignore[no-untyped-def]
        calls.append(part)
        return real(self, candidate_id, part)

    monkeypatch.setattr(loop_module.Loop, "part_item_scores", counted)
    loop = _run(tmp_path, project)
    seen = _state(loop)["SEEN"]
    assert seen and all(not s["val_item_scores"] for s in seen)
    assert loop._val_item_scores() == {} and calls == []


def test_item_scores_never_include_a_test_rollout(tmp_path):
    """A candidate scored on test (the final evaluation) still hands the picker its
    val items only: a test rollout recorded first does not leak in."""
    project = _project(tmp_path)
    loop = _run(tmp_path, project)
    secret = loop.item_key_secret
    val_keys = {opaque_item_key(i, secret) for i in gepa_ops.VAL_IDS}
    test_keys = {opaque_item_key(i, secret) for i in gepa_ops.TEST_IDS}
    before = loop._val_item_scores()
    for cid in before:
        # A test rollout with its own scores, put first among the candidate's parts.
        results = loop._part_results.setdefault(cid, {})
        test = [{"score": 0.5, "item_scores": [[i, 0.5] for i in gepa_ops.TEST_IDS]}]
        loop._part_results[cid] = {"test": test, **{k: v for k, v in results.items()
                                                    if k != "test"}}
    after = loop._val_item_scores()
    assert after == before and after
    for scores in after.values():
        assert set(scores) == val_keys and not set(scores) & test_keys


def test_run_part_item_scores_is_val_only():
    for part in ("test", "train"):
        with pytest.raises(OuterLoopError, match="'val' only"):
            Run(_loop=None).part_item_scores(None, part)  # type: ignore[arg-type]


# ── the library's table ──────────────────────────────────────────────────


def test_core_optimizers_are_registered_through_register():
    from refactory_dsh.optim import policies, random_optimizer

    expected = {
        "random": random_optimizer.RandomOptimizer,
        "autoresearch": random_optimizer.AutoresearchOptimizer,
        "archive": policies.ArchiveSelector, "evolutionary": policies.EvolutionarySelector,
        "islands": policies.IslandsSelector, "hill_climb": policies.HillClimbSelector,
        "map_elites": policies.MapElitesSelector,
    }
    assert {k: optim.REGISTRY[k] for k in expected} == expected
    # register refuses to silently rebind a taken name, and accepts the same class again.
    class Impostor(optim.Optimizer):
        name = "random"

    with pytest.raises(ValueError, match="already registered"):
        optim.register(Impostor)
    assert optim.register(random_optimizer.RandomOptimizer) is random_optimizer.RandomOptimizer
    with pytest.raises(TypeError):
        optim.register(object)  # type: ignore[arg-type]


def test_register_adds_a_selectable_optimizer(monkeypatch):
    monkeypatch.setattr(optim, "REGISTRY", dict(optim.REGISTRY))

    class Mine(optim.ParentSelector):
        name = "mine_by_hand"

    optim.register(Mine)
    assert optim.resolve("mine_by_hand") is Mine
    assert "mine_by_hand" in optim.picker_names()
    assert isinstance(optim.create("mine_by_hand", seed=0), Mine)


def test_a_package_optimizer_must_be_named_and_an_optimizer(tmp_path):
    class Nameless(optim.Optimizer):
        name = ""

    with pytest.raises(PackageError, match="optimizer class Nameless needs a class-level 'name'"):
        Package(name="bad", optimizers=(Nameless,))
    bad = SOURCE.replace("optimizers=(ProbePicker, ProbeProposer)",
                         "optimizers=(ProbePicker, ProbeProposer, NotOne)").replace(
        "SEEN = []", "class NotOne:\n    name = 'not_one'\n\n\nSEEN = []").replace(
        'name="probe_blocks"', 'name="probe_bad"')
    project = _project(tmp_path, bad, name="probe_bad")
    with pytest.raises(TypeError, match="not a Optimizer"):
        optim.package_optimizers(project)


def test_strategies_and_optimizers_share_one_discovery_route(tmp_path, monkeypatch):
    calls: list[str] = []
    real = registry.contributed

    def spy(attribute: str, base: type, project: Path | None = None,
            **kw: Any) -> dict[str, type]:
        calls.append(attribute)
        return real(attribute, base, project, **kw)

    monkeypatch.setattr(registry, "contributed", spy)
    strategies.package_strategies()
    optim.package_optimizers()
    assert calls == ["strategies", "optimizers"]


# ── settings, validation and selection defaults ────────────────────────────


def test_profiles_validate_a_projects_package_names(tmp_path):
    """A project's package strategy, optimizer and picker are valid settings for that
    project, and only for it: the installed tree alone does not have them."""
    project = _project(tmp_path)
    chosen = {"optimizer": "probe_proposer", "parent_picker": "probe_picker",
              "strategy": "probe_loop"}
    settings = profiles.normalize(dict(chosen), project)
    assert {k: settings[k] for k in chosen} == chosen
    for key in chosen:
        with pytest.raises(profiles.ProfileError, match=f"unknown {key}"):
            profiles.normalize({key: chosen[key]})
    with pytest.raises(profiles.ProfileError, match="unknown parent_picker"):
        profiles.normalize({"parent_picker": "probe_proposer"}, project)
    with pytest.raises(profiles.ProfileError, match="picks parents itself"):
        profiles.normalize({"optimizer": "probe_picker", "parent_picker": "archive"}, project)
    with pytest.raises(profiles.ProfileError, match="unknown optimizer"):
        profiles.normalize({"optimizer": "probe_nothing"}, project)
    # Saved and resolved as a profile of the project.
    profiles.save(project, "picks", {"settings": chosen})
    with _patch_mode(OneKnob):
        resolved = profiles.resolve(project, MODE_ID, name="picks")
    assert {k: resolved["settings"][k] for k in chosen} == chosen


def _cli(tmp_path: Path, project: Path, *extra: str, profile: bool = True) -> tuple[int, str]:
    """``python -m refactory_dsh.loop``, as a person runs it, in-process."""
    import contextlib
    import io

    from refactory_dsh import loop as loop_module

    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"train": gepa_ops.TRAIN_IDS, "val": gepa_ops.VAL_IDS}))
    if not (project / profiles.PROFILE_ROOT / "picks").exists():
        profiles.save(project, "picks", {"settings": {
            "parent_picker": "probe_picker", "parts_manifest": str(parts),
            "allowed_ops": ["set_knob"], "holdout": 0.0}})
    err = io.StringIO()
    with _patch_mode(OneKnob), contextlib.redirect_stderr(err), \
            contextlib.redirect_stdout(io.StringIO()):
        code = loop_module.main([MODE_ID, "--project", str(project),
                                 *(["--profile", "picks"] if profile else []),
                                 "--optimizer", "probe_proposer", "--epochs", "2",
                                 "--batch-size", "1", "--seed", "1",
                                 "--runs-root", str(tmp_path / "cli_run"), *extra])
    return code, err.getvalue()


def test_the_command_line_runs_a_projects_package_optimizers(tmp_path):
    project = _project(tmp_path)
    code, err = _cli(tmp_path, project)
    assert code == 0, err
    config = json.loads((tmp_path / "cli_run" / "config.json").read_text())
    assert (config["optimizer"], config["parent_picker"]) == ("probe_proposer", "probe_picker")
    source = {"package": PACKAGE, "version": "1.0.0"}
    assert config["package_sources"] == {
        "optimizer": {"name": "probe_proposer", **source},
        "parent_picker": {"name": "probe_picker", **source}}


def test_a_resume_refuses_another_version_of_a_package_optimizer(tmp_path):
    project = _project(tmp_path)
    code, err = _cli(tmp_path, project)
    assert code == 0, err
    init = registry.workspace_root(project) / PACKAGE / "__init__.py"
    # A different length too: an import cache keyed on size and mtime must not hide it.
    init.write_text(init.read_text().replace('version="1.0.0"', 'version="2.0.10"'))
    # Resumed as a person does, from the run's own record (no --profile).
    code, err = _cli(tmp_path, project, "--resume", "--epochs", "3", profile=False)
    assert code == 2
    assert ("optimizer was probe_proposer from package probe_blocks 1.0.0 and is now "
            "probe_proposer from package probe_blocks 2.0.10") in err
    # The recorded version resumes.
    init.write_text(init.read_text().replace('version="2.0.10"', 'version="1.0.0"'))
    code, err = _cli(tmp_path, project, "--resume", "--epochs", "3", profile=False)
    assert code == 0, err


# ── which packages contribute: the enabled ones, as for modes ──────────────


def test_a_disabled_package_contributes_nothing(tmp_path):
    project = _project(tmp_path)
    assert optim.resolve("probe_picker", project)
    assert strategies.resolve("probe_loop", project)
    registry.set_enabled(project, PACKAGE, False)
    with pytest.raises(ValueError, match="unknown optimizer"):
        optim.resolve("probe_picker", project)
    with pytest.raises(ValueError, match="unknown strategy"):
        strategies.resolve("probe_loop", project)
    with pytest.raises(profiles.ProfileError, match="unknown parent_picker"):
        profiles.normalize({"parent_picker": "probe_picker"}, project)
    # As a mode of a disabled package is refused (`registry.resolve`).
    with pytest.raises(registry.PackageError, match="disabled"):
        registry.resolve(project, f"{PACKAGE}/anything")
    registry.set_enabled(project, PACKAGE, True)
    assert optim.resolve("probe_picker", project)


def test_a_generated_package_contributes_as_its_modes_run(tmp_path):
    """``registry.add`` records a package as ``generated``. Its modes resolve and run
    (``registry.resolve`` checks only that the package is enabled), so its
    optimizers and strategies resolve too: one policy for every piece of a package."""
    source = tmp_path / "src" / "added_blocks"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(SOURCE.replace('name="probe_blocks"',
                                                       'name="added_blocks"'))
    project = tmp_path / "project"
    project.mkdir()
    registry.add(project, source)
    assert registry.load("added_blocks", project).trust == "generated"
    assert optim.resolve("probe_picker", project).__module__ == "refactory_packages.added_blocks"
    assert strategies.resolve("probe_loop", project).name == "probe_loop"
    registry.set_enabled(project, "added_blocks", False)
    with pytest.raises(ValueError, match="unknown optimizer"):
        optim.resolve("probe_picker", project)


# ── names are never resolved silently ──────────────────────────────────────


@pytest.mark.parametrize("shadow,attribute", [("archive", "optimizers"), ("random", "optimizers"),
                                              ("default", "strategies"),
                                              ("reflective", "strategies")])
def test_a_package_name_that_shadows_the_librarys_is_refused(tmp_path, shadow, attribute):
    renamed = SOURCE.replace('name = "probe_picker"' if attribute == "optimizers"
                             else 'name = "probe_loop"', f'name = "{shadow}"')
    project = _project(tmp_path, renamed)
    with pytest.raises(registry.PackageError, match=f"{shadow!r}, a name the library already has"):
        (optim.package_optimizers if attribute == "optimizers"
         else strategies.package_strategies)(project)
    with pytest.raises(registry.PackageError, match="already has"):
        (optim.available if attribute == "optimizers" else strategies.available)(project)
    # The library's name still means the library's class.
    if attribute == "optimizers":
        assert optim.resolve(shadow, project) is optim.REGISTRY[shadow]
    else:
        assert strategies.resolve(shadow, project) is strategies.STRATEGIES[shadow]


def test_two_packages_with_one_name_are_refused(tmp_path):
    project = _project(tmp_path)
    twin = registry.workspace_root(project) / "probe_twin"
    twin.mkdir()
    (twin / "__init__.py").write_text(SOURCE.replace('name="probe_blocks"', 'name="probe_twin"'))
    with pytest.raises(registry.PackageError, match="both list 'probe_"):
        optim.resolve("probe_picker", project)
    with pytest.raises(registry.PackageError, match="both list 'probe_loop'"):
        strategies.resolve("probe_loop", project)


def test_a_package_population_picker_defaults_to_the_pool(tmp_path):
    project = _project(tmp_path)
    # ProbePicker keeps ParentSelector's population=True, as archive does.
    assert default_selection("default", "random", "probe_picker", project) == "pool"
    assert default_selection("default", "random", "archive") == "pool"
    assert default_selection("default", "random", "hill_climb") == "checkpoint"
    assert default_selection("default", "random", "") == "checkpoint"


def test_help_names_package_optimizers():
    assert "package" in profiles.SETTING_HELP["parent_picker"]
    assert "package" in profiles.CLI_SETTINGS["optimizer"]


def test_the_catalogue_lists_a_projects_package_pieces(tmp_path):
    from refactory_dsh.console import level_two

    project = _project(tmp_path)
    listed = level_two(project)
    optimizers = {e["name"] for e in listed["optimizers"]}
    assert {"probe_picker", "probe_proposer", "archive", "random"} <= optimizers
    assert {"probe_loop", "default"} <= {e["name"] for e in listed["outer_loops"]}
    assert "probe_picker" not in {e["name"] for e in level_two()["optimizers"]}
