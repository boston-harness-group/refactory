"""Parent pickers and proposers as package building blocks (``Package.optimizers``).

A package lists optimizers the way it lists strategies, and a run selects them by name
through the normal settings (``optimizer``, ``parent_picker``). Core's own optimizers
go through the same ``optim.register``. A picker is handed each judged candidate's
per-item validation scores (``ProposalContext.val_item_scores``), keyed by the run's
HMAC of the item id, never a raw id and never test. Each test fails when the rule it
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

    SEEN = []
    PROPOSED = []
    TEXTS = ["Always answer ALPHA.", "Always answer BETA.", "Say ALPHA.", "Say BETA."]


    class ProbePicker(ParentSelector):
        """Records the per-item val scores it is handed; picks the best by val."""

        name = "probe_picker"
        policy = "the probe's pick"

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
            PROPOSED.append({"parents": [p.id for p in parents], "text": text})
            return [Proposal(edit={"op": "set_knob", "name": "instruction", "value": text},
                             parent=parents[0].id)]


    PACKAGE = Package(name="probe_blocks", description="probe optimizers",
                      optimizers=(ProbePicker, ProbeProposer))
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


def test_run_part_item_scores_is_val_only():
    for part in ("test", "train"):
        with pytest.raises(OuterLoopError, match="'val' only"):
            Run(_loop=None).part_item_scores(None, part)  # type: ignore[arg-type]


# ── one registration path ──────────────────────────────────────────────────


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

    def spy(attribute: str, base: type, project: Path | None = None) -> dict[str, type]:
        calls.append(attribute)
        return real(attribute, base, project)

    monkeypatch.setattr(registry, "contributed", spy)
    strategies.package_strategies()
    optim.package_optimizers()
    assert calls == ["strategies", "optimizers"]


# ── settings, validation and selection defaults ────────────────────────────


def test_profiles_validate_package_optimizers(monkeypatch, tmp_path):
    project = _project(tmp_path)
    found = registry.discover(project)
    monkeypatch.setattr(registry, "discover", lambda project=None: found)
    settings = profiles.normalize({"optimizer": "probe_proposer", "parent_picker": "probe_picker"})
    assert (settings["optimizer"], settings["parent_picker"]) == ("probe_proposer", "probe_picker")
    with pytest.raises(profiles.ProfileError, match="unknown parent_picker"):
        profiles.normalize({"parent_picker": "probe_proposer"})
    with pytest.raises(profiles.ProfileError, match="picks parents itself"):
        profiles.normalize({"optimizer": "probe_picker", "parent_picker": "archive"})
    with pytest.raises(profiles.ProfileError, match="unknown optimizer"):
        profiles.normalize({"optimizer": "probe_nothing"})


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
