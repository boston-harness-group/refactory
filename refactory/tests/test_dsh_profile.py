"""The DSH profiles this checkout composes are machine state, and a gap is legible.

Issue #45: `dsh plugin --profile graph add <package>` creates the profile, nothing
in the install path did, and the first mode run on DSH stopped with a message about
a missing *profile* that reads as a packaging bug. These assert the three states a
caller has to tell apart -- absent, present without a layer, ready -- and that a
run refuses before spawning rather than letting the child produce the bare error.

Issue #277: the install path appended re:factory's bundles to the harness's shipped
`web` profile. These also assert the redesign: the GUI profile is a re:factory-owned
`refactory` initialized from the `web` template, the harness's `web` is never a
target, and the plan `install.sh`, the launcher and `doctor` share is emitted from
one table rather than a second copy in shell.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from refactory_dsh import dsh_profile, registry
from refactory_dsh.dsh_runtime import run_on_dsh
from refactory_dsh.paths import REPO_ROOT
from tests_support import _link_bundle, _profile

#: The preset bundle this checkout renamed. Spelled out because a profile manifest
#: written before the rename carries exactly this string, and the migration tests
#: are about that manifest rather than about the table that names it.
_RETIRED_PRESETS = "@deepseek-ai/dsh-refactory-presets"

#: The profile name this checkout used to compose, from the table that retires it.
_RETIRED_PROFILE = next(iter(dsh_profile.RETIRED_PROFILES))



@pytest.fixture(autouse=True)
def _fixed_node(monkeypatch):
    """These tests fake `subprocess.Popen` to watch DSH being spawned. Choosing a Node
    may run `node --version`, which would reach the fake, so the choice is fixed."""
    monkeypatch.setenv("NODE_BIN", "node")

def _declare(directory: Path, packages: list[str]) -> None:
    """Write ``dependencies`` entries for ``packages``, as `dsh plugin add` would.

    The two lists in a profile manifest are independent, and the plan has to read
    both: `pnpm remove` speaks ``dependencies``, and the harness composes from
    ``dsh.profile.bundles``.
    """
    manifest_path = directory / "package.json"
    manifest = json.loads(manifest_path.read_text())
    dependencies = manifest.setdefault("dependencies", {})
    for package in packages:
        dependencies[package] = "link:/checkout"
    manifest_path.write_text(json.dumps(manifest))


def _apply_plan(home: Path, steps: list[dsh_profile.ProvisionStep]) -> None:
    """Model plan rows for unit tests, not harness reconciliation.

    * ``dsh plugin remove`` forwards to ``pnpm remove``, which edits only
      ``dependencies`` and never ``dsh.profile.bundles``. It also removes nothing
      at all when any named package is not declared there, so a row naming an
      undeclared package fails -- which is what ``set -e`` turns into an aborted
      plan.
    * ``add`` writes the dependency and the harness appends the bundle.
    * ``init`` copies the shipped template's bundle list.
    * ``clean`` is the manifest rewrite: it drops the named selections from
      ``dsh.profile.bundles`` and touches nothing else.

    Real launcher and lock behavior are covered by test_dsh_profile_migration.py.
    """
    for step in steps:
        if step.kind == "init":
            _profile(home, step.profile, ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"])
            continue
        manifest_path = home / "profiles" / step.profile / "package.json"
        manifest = json.loads(manifest_path.read_text())
        names = step.value.split()
        dependencies = manifest.setdefault("dependencies", {})
        bundle_list = manifest["dsh"]["profile"]["bundles"]
        if step.kind == "remove":
            undeclared = [name for name in names if name not in dependencies]
            assert not undeclared, f"pnpm remove would remove nothing, failing on {undeclared}"
            for name in names:
                del dependencies[name]
        elif step.kind == "clean":
            manifest["dsh"]["profile"]["bundles"] = [
                bundle for bundle in bundle_list if bundle not in names
            ]
        else:
            package = f"@deepseek-ai/dsh-{step.value}"
            dependencies[package] = "link:/checkout"
            if package not in bundle_list:
                bundle_list.append(package)
        manifest_path.write_text(json.dumps(manifest))


def _ready(home: Path) -> None:
    """Every owned target ready: graph and a fully layered refactory."""
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    _profile(
        home,
        dsh_profile.GUI_PROFILE,
        [
            "@deepseek-ai/dsh-base",
            "@deepseek-ai/dsh-web-app",
            dsh_profile.PRESETS_BUNDLE,
            dsh_profile.WEB_SURFACE_BUNDLE,
        ],
    )


def test_a_missing_profile_names_the_command_and_the_reason(tmp_path, monkeypatch):
    """'does not exist' is not actionable; the command and why are."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    problem = dsh_profile.requirement()
    assert problem is not None
    assert "machine state" in problem, "the reason the profile is absent is not stated"
    assert dsh_profile.add_command() in problem, "the fix is not copy-pasteable"


def test_the_add_command_names_the_bundle_in_the_checkout():
    command = dsh_profile.add_command()
    assert command.startswith("dsh plugin --profile graph add ")
    assert command.endswith(str(REPO_ROOT / "packages" / "bundle" / "graph"))


def test_a_profile_without_the_bundle_is_a_different_problem(tmp_path, monkeypatch):
    """Present and unready is not the same failure as absent."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base"])
    monkeypatch.setenv("DSH_HOME", str(home))
    problem = dsh_profile.requirement()
    assert problem is not None
    assert dsh_profile.GRAPH_BUNDLE in problem
    assert "does not exist" not in problem


def test_a_profile_with_the_bundle_is_ready(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.is_ready()
    assert dsh_profile.requirement() is None
    dsh_profile.require()


def test_a_blank_dsh_home_falls_back_to_the_default(monkeypatch):
    """Whitespace-only is unset, not a directory named ' '."""
    monkeypatch.setenv("DSH_HOME", "   ")
    assert dsh_profile.dsh_home() == Path.home() / ".dsh"


def test_the_gui_profile_names_the_preset_bundle_not_a_directory_called_web():
    """Each layer's directory is derived from the bundle NAME, not the profile name.

    The GUI profile is `refactory`; its layers are `packages/bundle/refactory` and
    `packages/bundle/refactory-web`. Deriving a directory from the profile name
    would make both layers the same one, which is the bug the derivation exists to
    prevent.
    """
    command = dsh_profile.add_command(dsh_profile.GUI_PROFILE)
    assert command.startswith("dsh plugin --profile refactory add ")
    assert command.endswith(str(REPO_ROOT / "packages" / "bundle" / "refactory"))
    surface = dsh_profile.add_command(dsh_profile.GUI_PROFILE, dsh_profile.WEB_SURFACE_BUNDLE)
    assert surface.endswith(str(REPO_ROOT / "packages" / "bundle" / "refactory-web"))


def test_the_gui_profile_initializes_from_the_harness_web_template():
    """The owned GUI profile copies the template; it does not replace it.

    `dsh plugin add` alone would initialize a bare profile with none of the
    harness's `web` bundles, so the template init has to come first and name the
    shipped template. The harness's own `web` profile is never written to.
    """
    command = dsh_profile.init_command(dsh_profile.GUI_PROFILE)
    assert command == "dsh --profile refactory --from-default-profile web --dump-config"


def test_a_template_backed_target_without_a_template_initializes_from_add():
    """`graph` has no shipped template, so its first `dsh plugin add` creates it."""
    assert dsh_profile.init_command(dsh_profile.GRAPH_PROFILE) == dsh_profile.add_command(
        dsh_profile.GRAPH_PROFILE
    )


def test_the_harness_web_template_is_not_a_target():
    """The profile the old install mutated is not composed by this checkout."""
    assert dsh_profile.WEB_TEMPLATE not in {entry.profile for entry in dsh_profile.TARGETS}
    with pytest.raises(KeyError):
        dsh_profile.add_command(dsh_profile.WEB_TEMPLATE)


def test_a_missing_owned_profile_names_the_template_init(tmp_path, monkeypatch):
    """Absent-and-template-backed is fixed by `init` first, then the layer add."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    problem = dsh_profile.requirement(dsh_profile.GUI_PROFILE, dsh_profile.PRESETS_BUNDLE)
    assert problem is not None
    assert dsh_profile.init_command(dsh_profile.GUI_PROFILE) in problem
    assert "--from-default-profile web" in problem


def test_an_unowned_profile_is_an_error_not_a_guess():
    """A profile this checkout does not own must not borrow another one's command.

    Falling back to `graph` would print a copy-pasteable command naming the wrong
    bundle directory: it reads as a fix, and running it creates a profile that
    silently does not carry the bundle the caller asked for.
    """
    with pytest.raises(KeyError):
        dsh_profile.add_command("integration")
    with pytest.raises(KeyError):
        # A bundle the profile does not compose is refused too, not turned into a path.
        dsh_profile.add_command(dsh_profile.GUI_PROFILE, dsh_profile.GRAPH_BUNDLE)
    assert dsh_profile.bundle_directory(dsh_profile.PRESETS_BUNDLE) == "refactory"
    assert dsh_profile.bundle_directory(dsh_profile.WEB_SURFACE_BUNDLE) == "refactory-web"
    assert dsh_profile.bundle_directory(dsh_profile.GRAPH_BUNDLE) == "graph"


def test_the_gui_target_composes_both_of_its_layers():
    """The preset and the surface are separate layers, and both are installed."""
    presets = dsh_profile.add_command(dsh_profile.GUI_PROFILE, dsh_profile.PRESETS_BUNDLE)
    surface = dsh_profile.add_command(dsh_profile.GUI_PROFILE, dsh_profile.WEB_SURFACE_BUNDLE)
    assert presets.endswith(str(REPO_ROOT / "packages" / "bundle" / "refactory"))
    assert surface.endswith(str(REPO_ROOT / "packages" / "bundle" / "refactory-web"))
    assert dsh_profile.add_command(dsh_profile.GUI_PROFILE) == presets, (
        "omitting the layer keeps the long-standing command for the first layer"
    )


def test_the_gui_profile_without_the_preset_layer_is_reported(tmp_path, monkeypatch):
    """A profile that exists but lacks the preset layer is a gap, not ready.

    This is the state a GUI lands in when the layer is never installed: the
    composition still names `ceo` as the roster default and the profile has no
    `ceo` preset, so every session created without naming one stops.
    """
    home = tmp_path / "home"
    _profile(home, dsh_profile.GUI_PROFILE, ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"])
    monkeypatch.setenv("DSH_HOME", str(home))
    problem = dsh_profile.requirement(dsh_profile.GUI_PROFILE, dsh_profile.PRESETS_BUNDLE)
    assert problem is not None
    assert dsh_profile.PRESETS_BUNDLE in problem
    assert not dsh_profile.is_ready(dsh_profile.GUI_PROFILE, dsh_profile.PRESETS_BUNDLE)


def test_the_gui_profile_without_the_surface_layer_is_reported(tmp_path, monkeypatch):
    """The preset alone is not a ready GUI profile.

    Without the surface layer the console and picker rows never mount, so the GUI
    boots with no refactory panel even though the `ceo` preset resolves. The gap
    must name that bundle and the command that installs exactly it.
    """
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    _profile(
        home,
        dsh_profile.GUI_PROFILE,
        ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app", dsh_profile.PRESETS_BUNDLE],
    )
    monkeypatch.setenv("DSH_HOME", str(home))
    gaps = dsh_profile.missing_bundles()
    assert [(profile, bundle) for profile, bundle, _why in gaps] == [
        (dsh_profile.GUI_PROFILE, dsh_profile.WEB_SURFACE_BUNDLE)
    ], gaps
    assert dsh_profile.add_command(
        dsh_profile.GUI_PROFILE, dsh_profile.WEB_SURFACE_BUNDLE
    ) in gaps[0][2]


def test_missing_bundles_reports_each_owned_target_that_is_short(tmp_path, monkeypatch):
    """Ready targets stay out, and every owned layer a target needs is a row."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    gaps = dsh_profile.missing_bundles()
    assert [(profile, bundle) for profile, bundle, _why in gaps] == [
        (dsh_profile.GUI_PROFILE, dsh_profile.PRESETS_BUNDLE),
        (dsh_profile.GUI_PROFILE, dsh_profile.WEB_SURFACE_BUNDLE),
    ], gaps


def test_missing_bundles_is_empty_when_every_owned_target_is_ready(tmp_path, monkeypatch):
    """Ready means no work: `install.sh` and the launcher both branch on this."""
    home = tmp_path / "home"
    _ready(home)
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.missing_bundles() == []


def test_the_provision_plan_initializes_the_template_before_adding_layers(tmp_path, monkeypatch):
    """A fresh machine gets the template init before the GUI layers, graph after.

    The order is the plan's contract: `dsh plugin add` on a profile with no
    directory initializes a *bare* profile, so the template init has to precede
    the layer adds or `dsh-web-app` would never be listed and the console and
    picker rows would not mount.
    """
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    plan = dsh_profile.provision_plan()
    assert [tuple(step) for step in plan] == [
        ("add", "graph", "graph"),
        ("init", "refactory", "web"),
        ("add", "refactory", "refactory"),
        ("add", "refactory", "refactory-web"),
    ], plan
    assert dsh_profile.render_plan(plan).splitlines() == [
        "add\tgraph\tgraph",
        "init\trefactory\tweb",
        "add\trefactory\trefactory",
        "add\trefactory\trefactory-web",
    ]


def test_the_provision_plan_is_empty_when_every_owned_target_is_ready(tmp_path, monkeypatch):
    """A provisioned machine must not be touched at boot."""
    home = tmp_path / "home"
    _ready(home)
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.provision_plan() == []
    assert dsh_profile.render_plan() == ""


def test_the_plan_retires_the_renamed_bundle_before_it_adds_the_replacement(
    tmp_path, monkeypatch
):
    """A profile carrying the pre-rename name is migrated, not left dangling.

    The removal precedes the adds, and it names only a package the manifest
    declares as a dependency: `pnpm remove` removes nothing at all when one of its
    names is missing from that list, which aborted the launcher's `set -e` script
    rather than healing anything. The retired name is a dependency here, so it is
    removed; the initial `clean` row clears obsolete selections before pnpm runs.
    """
    home = tmp_path / "home"
    _profile(home, dsh_profile.GRAPH_PROFILE, ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    legacy = _profile(
        home,
        dsh_profile.GUI_PROFILE,
        ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app", _RETIRED_PRESETS],
    )
    _declare(legacy, [_RETIRED_PRESETS])
    monkeypatch.setenv("DSH_HOME", str(home))
    steps = dsh_profile.provision_plan()
    assert [tuple(step) for step in steps] == [
        ("clean", dsh_profile.GUI_PROFILE, _RETIRED_PRESETS),
        ("remove", dsh_profile.GUI_PROFILE, _RETIRED_PRESETS),
        ("add", dsh_profile.GUI_PROFILE, "refactory"),
        ("add", dsh_profile.GUI_PROFILE, "refactory-web"),
    ], steps
    _apply_plan(home, steps)
    assert dsh_profile.declared_bundles(dsh_profile.GUI_PROFILE) == [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
        dsh_profile.PRESETS_BUNDLE,
        dsh_profile.WEB_SURFACE_BUNDLE,
    ]
    manifest = json.loads((legacy / "package.json").read_text())
    assert _RETIRED_PRESETS not in manifest["dependencies"], manifest
    assert dsh_profile.provision_plan() == [], "the migration is not idempotent"


def test_the_plan_clears_a_retired_name_that_is_no_longer_a_dependency(
    tmp_path, monkeypatch
):
    """The entry no command can reach is cleared by the manifest rewrite.

    This is the state that never converged: the retired name is still selected but
    is no longer a dependency, so `pnpm remove` cannot name it and the harness's
    `reconcile()` keeps rather than drops it. No `remove` row is emitted -- one
    would remove nothing and abort the plan -- and the `clean` row is what clears
    it.
    """
    home = tmp_path / "home"
    _profile(home, dsh_profile.GRAPH_PROFILE, ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    legacy = _profile(
        home,
        dsh_profile.GUI_PROFILE,
        ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app", _RETIRED_PRESETS],
    )
    monkeypatch.setenv("DSH_HOME", str(home))
    steps = dsh_profile.provision_plan()
    assert [step.kind for step in steps] == ["clean", "add", "add"], steps
    _apply_plan(home, steps)
    assert dsh_profile.declared_bundles(dsh_profile.GUI_PROFILE) == [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
        dsh_profile.PRESETS_BUNDLE,
        dsh_profile.WEB_SURFACE_BUNDLE,
    ]
    assert dsh_profile.provision_plan() == [], "the cleanup is not idempotent"


def test_the_plan_repairs_a_wanted_layer_that_lost_its_dependency(
    tmp_path, monkeypatch
):
    """A wanted layer missing from `dependencies` is a missing install, not junk.

    Provisioning must restore the dependency of an already-selected layer.
    """
    home = tmp_path / "home"
    _profile(home, dsh_profile.GRAPH_PROFILE, ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    gui = _profile(
        home,
        dsh_profile.GUI_PROFILE,
        [
            "@deepseek-ai/dsh-base",
            "@deepseek-ai/dsh-web-app",
            dsh_profile.PRESETS_BUNDLE,
            dsh_profile.WEB_SURFACE_BUNDLE,
        ],
    )
    manifest_path = gui / "package.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["dependencies"][dsh_profile.PRESETS_BUNDLE]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.obsolete_selections(dsh_profile.GUI_PROFILE) == [], (
        "a wanted layer must never be treated as an obsolete selection"
    )
    steps = dsh_profile.provision_plan()
    assert [tuple(step) for step in steps] == [
        ("add", dsh_profile.GUI_PROFILE, "refactory"),
    ], steps
    _apply_plan(home, steps)
    manifest = json.loads((gui / "package.json").read_text())
    assert dsh_profile.PRESETS_BUNDLE in manifest["dependencies"], manifest
    assert dsh_profile.declared_bundles(dsh_profile.GUI_PROFILE) == [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
        dsh_profile.PRESETS_BUNDLE,
        dsh_profile.WEB_SURFACE_BUNDLE,
    ]
    assert dsh_profile.provision_plan() == []


def test_retirement_resumes_after_cleanup_before_dependency_removal(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _ready(home)
    stranded = _profile(home, _RETIRED_PROFILE, ["@deepseek-ai/dsh-base"])
    _declare(stranded, [_RETIRED_PRESETS])
    monkeypatch.setenv("DSH_HOME", str(home))
    assert [tuple(step) for step in dsh_profile.provision_plan()] == [
        ("remove", _RETIRED_PROFILE, _RETIRED_PRESETS),
    ]


@pytest.mark.parametrize("name", ["web", "martty", "operator", "../web"])
def test_cleanup_refuses_unowned_profiles(tmp_path, monkeypatch, name):
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    with pytest.raises(dsh_profile.ProfileError, match="does not own"):
        dsh_profile.clean_manifest(name)


def test_the_plan_retires_the_layers_from_a_profile_the_rename_stranded(
    tmp_path, monkeypatch
):
    """The old directory stays, but re:factory no longer has a layer in it.

    A profile cannot be moved: its `node_modules` holds absolute symlinks under
    `.dsh-module-fallback`, so the plan creates the profile under its new name and
    clears re:factory's layers out of the old one. What is left is an inert copy
    of the `web` template, which is what stops the pre-rename bundle entry from
    dangling on a directory this checkout no longer has.

    A stranded profile gets no `add` rows, so no plugin operation ever runs there
    to trigger the harness's own reconcile: the `clean` row is the only mechanism
    that can reach these entries.
    """
    home = tmp_path / "home"
    _profile(home, dsh_profile.GRAPH_PROFILE, ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    stranded = _profile(
        home,
        _RETIRED_PROFILE,
        [
            "@deepseek-ai/dsh-base",
            "@deepseek-ai/dsh-web-app",
            _RETIRED_PRESETS,
            dsh_profile.WEB_SURFACE_BUNDLE,
        ],
    )
    _declare(stranded, [_RETIRED_PRESETS, dsh_profile.WEB_SURFACE_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    steps = dsh_profile.provision_plan()
    assert [tuple(step) for step in steps] == [
        ("init", dsh_profile.GUI_PROFILE, "web"),
        ("add", dsh_profile.GUI_PROFILE, "refactory"),
        ("add", dsh_profile.GUI_PROFILE, "refactory-web"),
        ("clean", _RETIRED_PROFILE, f"{_RETIRED_PRESETS} {dsh_profile.WEB_SURFACE_BUNDLE}"),
        ("remove", _RETIRED_PROFILE, f"{_RETIRED_PRESETS} {dsh_profile.WEB_SURFACE_BUNDLE}"),
    ], steps
    _apply_plan(home, steps)
    assert dsh_profile.declared_bundles(_RETIRED_PROFILE) == [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
    ]
    manifest = json.loads((stranded / "package.json").read_text())
    assert manifest["dependencies"] == {}, manifest
    assert dsh_profile.provision_plan() == [], "the retirement is not idempotent"


def test_doctor_names_the_removal_a_retired_bundle_needs_on_an_attached_profile(
    tmp_path, monkeypatch
):
    """re:factory never mutates Martty, so there the migration is only a command."""
    home = tmp_path / "home"
    _ready(home)
    _profile(home, dsh_profile.MARTTY_PROFILE, ["@deepseek-ai/dsh-base", _RETIRED_PRESETS])
    monkeypatch.setenv("DSH_HOME", str(home))
    gaps = dsh_profile.readiness_gaps()
    assert [(profile, bundle) for profile, bundle, _why in gaps] == [
        (dsh_profile.MARTTY_PROFILE, _RETIRED_PRESETS),
        (dsh_profile.MARTTY_PROFILE, dsh_profile.PRESETS_BUNDLE),
    ], gaps
    assert dsh_profile.migration_command(
        dsh_profile.MARTTY_PROFILE, [_RETIRED_PRESETS]
    ) in gaps[0][2]
    assert all(
        step.profile != dsh_profile.MARTTY_PROFILE for step in dsh_profile.provision_plan()
    ), "the plan must not remove a layer from a profile re:factory does not own"


def test_readiness_gaps_reports_an_installed_attached_target_but_not_an_absent_one(
    tmp_path, monkeypatch
):
    """An attached target is the operator's; a profile that is absent is not a gap.

    re:factory does not create or mutate a profile it does not own, so an absent
    Martty profile must not read as a broken install while an installed one that
    lacks the layer must be reported with the command that adds it.
    """
    home = tmp_path / "home"
    _ready(home)
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.readiness_gaps() == []
    _profile(home, dsh_profile.MARTTY_PROFILE, ["@deepseek-ai/dsh-base"])
    gaps = dsh_profile.readiness_gaps()
    assert [(profile, bundle) for profile, bundle, _why in gaps] == [
        (dsh_profile.MARTTY_PROFILE, dsh_profile.PRESETS_BUNDLE)
    ]
    assert dsh_profile.add_command(
        dsh_profile.MARTTY_PROFILE, dsh_profile.PRESETS_BUNDLE
    ) in gaps[0][2]
    assert all(
        step.profile != dsh_profile.MARTTY_PROFILE for step in dsh_profile.provision_plan()
    ), "install.sh must not create an attached profile"

    # An attached profile that is *absent* is not a gap at all, and asking about
    # it directly must say who owns creating it rather than falling back to a
    # copy-pasteable command that would create the operator's stack for them.
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "empty"))
    assert all(
        profile != dsh_profile.MARTTY_PROFILE
        for profile, _bundle, _why in dsh_profile.readiness_gaps()
    ), "an absent attached profile is not a gap"
    absent = dsh_profile.requirement(dsh_profile.MARTTY_PROFILE, dsh_profile.PRESETS_BUNDLE)
    assert absent is not None
    assert "does not create a profile it does not own" in absent


def test_stale_mutations_finds_a_harness_profile_an_old_install_mutated(tmp_path, monkeypatch):
    """The migration is stated, not performed: old installs polluted `web`."""
    home = tmp_path / "home"
    _profile(
        home,
        dsh_profile.WEB_TEMPLATE,
        [
            "@deepseek-ai/dsh-base",
            "@deepseek-ai/dsh-web-app",
            dsh_profile.PRESETS_BUNDLE,
            dsh_profile.WEB_SURFACE_BUNDLE,
        ],
    )
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.stale_mutations() == [
        (dsh_profile.WEB_TEMPLATE, [dsh_profile.PRESETS_BUNDLE, dsh_profile.WEB_SURFACE_BUNDLE])
    ]
    assert dsh_profile.migration_command(dsh_profile.WEB_TEMPLATE, [dsh_profile.PRESETS_BUNDLE]) == (
        f"dsh plugin --profile web remove {dsh_profile.PRESETS_BUNDLE}"
    )
    pristine = tmp_path / "pristine"
    _profile(pristine, dsh_profile.WEB_TEMPLATE, ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"])
    monkeypatch.setenv("DSH_HOME", str(pristine))
    assert dsh_profile.stale_mutations() == [], "a pristine template is not a mutation"


def test_stale_mutations_names_the_retired_bundle_the_migration_command_must_clear(
    tmp_path, monkeypatch
):
    """`doctor` names every re:factory name a polluted template carries.

    The retired name is the one that matters: it is the bundle whose directory no
    longer exists, so a command naming only the current one leaves that dangling
    entry declared and the report's own fix cannot clear what the report is about.
    """
    home = tmp_path / "home"
    _profile(
        home,
        dsh_profile.WEB_TEMPLATE,
        [
            "@deepseek-ai/dsh-base",
            "@deepseek-ai/dsh-web-app",
            _RETIRED_PRESETS,
            dsh_profile.WEB_SURFACE_BUNDLE,
        ],
    )
    monkeypatch.setenv("DSH_HOME", str(home))
    assert dsh_profile.stale_mutations() == [
        (dsh_profile.WEB_TEMPLATE, [_RETIRED_PRESETS, dsh_profile.WEB_SURFACE_BUNDLE])
    ]
    # The command has to name both: the retired name is missing from `LAYERS`, so
    # a set built from that table alone drops exactly the entry that dangles.
    assert dsh_profile.migration_command(
        dsh_profile.WEB_TEMPLATE, dsh_profile.stale_mutations()[0][1]
    ) == (
        f"dsh plugin --profile web remove {_RETIRED_PRESETS} "
        f"{dsh_profile.WEB_SURFACE_BUNDLE}"
    )


class _FakeProcess:
    """The slice of ``Popen`` the adapter uses: streamed lines and ``wait``.

    ``run_on_dsh`` streams the child through ``Popen`` so a timeout can shut down
    its owned tool processes; a fake built for ``subprocess.run`` no longer stands
    in for the spawn.
    """

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self._returncode = returncode
        self.stdout = iter(stdout.splitlines(keepends=True))
        self.stderr = iter(stderr.splitlines(keepends=True))
        self.returncode: int | None = None

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._returncode
        return self._returncode


def test_run_on_dsh_hands_the_runner_an_absolute_workdir(tmp_path, monkeypatch):
    """DSH refuses a relative session cwd, so the adapter must not pass one on.

    The CLI's default run root is relative, which is how a whole `dsh`-evaluated
    run came to fail every candidate with "session header cwd must be an absolute
    path" while the same state scored under the walker.
    """
    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    seen: dict[str, list[str]] = {}

    def _capture(command, **kwargs):  # noqa: ANN001, ANN003 - the runner's own call
        seen["command"] = command
        return _FakeProcess(1, stderr="dsh: session header cwd must be an absolute path")

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", _capture)
    run_on_dsh(state, Path("relative/run"), name="probe")
    argv = seen["command"]
    workdir = argv[argv.index("--workdir") + 1]
    assert Path(workdir).is_absolute(), f"DSH refuses a relative cwd: {workdir!r}"


def test_run_on_dsh_hands_the_runner_the_knob_values_once(tmp_path, monkeypatch):
    """`--knobs` is the knobs JSON, not a JSON string that contains it.

    Double-encoding it makes the node runner fall back to each op's default, so
    DSH runs a different program than the candidate the loop scored.
    """
    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    seen: dict[str, list[str]] = {}

    def _capture(command, **kwargs):  # noqa: ANN001, ANN003 - the runner's own call
        seen["command"] = command
        return _FakeProcess(1, stderr="probe")

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", _capture)
    run_on_dsh(state, tmp_path / "run", name="probe")
    argv = seen["command"]
    knobs = json.loads(argv[argv.index("--knobs") + 1])
    assert isinstance(knobs, dict), f"--knobs must carry the values, got {knobs!r}"
    assert knobs == {key: entry["value"] for key, entry in state["knobs"].items()}


def test_run_on_dsh_hands_each_output_line_on_while_keeping_the_parse(tmp_path, monkeypatch):
    """A watcher sees the child while it runs, and `best_score` still parses.

    Teeing is not a replacement for capturing: the runner's last JSON object is
    where the score comes from, so both have to happen from one read of the child's
    output.
    """
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)
    script = tmp_path / "stub_dsh.py"
    script.write_text(
        "import json\n"
        "print('node start')\n"
        "print('node done')\n"
        "print(json.dumps({'best_score': 0.75}))\n"
    )
    monkeypatch.setattr(
        "refactory_dsh.dsh_runtime.dsh_command",
        lambda **_: [sys.executable, str(script)],
    )
    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    seen: list[str] = []
    run = run_on_dsh(state, tmp_path / "run", name="probe", on_output=seen.append)
    assert run.best_score == 0.75, "the runner's own JSON is still what scores the run"
    assert any("node start" in line for line in seen), seen
    assert "node done" in run.stdout
    assert run.stderr == ""


def test_run_on_dsh_refuses_before_spawning_when_the_profile_is_absent(
    tmp_path, monkeypatch
):
    """The runner checks first, so the error is the actionable one, not the child's."""
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)

    def _no_spawn(*args, **kwargs):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("a subprocess was spawned without a usable profile")

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", _no_spawn)
    run = run_on_dsh({}, tmp_path / "run")
    assert run.returncode == 2
    assert dsh_profile.add_command() in run.stderr
    assert "machine state" in run.stderr


@pytest.mark.parametrize("session,worker,explicit,expected", [
    ("session", "worker", None, "worker"),
    ("session", None, None, "session"),
    (None, "worker", None, "worker"),
    (None, None, None, "user/.dsh"),
    ("   ", " \t ", None, "user/.dsh"),
    ("session", " \t ", None, "session"),
    ("session", "worker", "explicit", "explicit"),
    ("session", "worker", "   ", "worker"),
    ("session", "~/worker", None, "user/worker"),
    ("session", "worker", "relative", "relative"),
])
def test_model_home_preflight_and_child_agree(tmp_path, monkeypatch, session, worker, explicit, expected):
    from refactory_dsh.model import DshModelClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    for key, value in [("DSH_HOME", session), ("REFACTORY_MODEL_DSH_HOME", worker)]:
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value if value.isspace() or value.startswith("~") else str(tmp_path / value))
    chosen = tmp_path / expected
    # The session home is empty: only the chosen home has a ready graph profile.
    _profile(chosen, "graph", [dsh_profile.GRAPH_BUNDLE])
    client = DshModelClient(dsh_home=explicit if explicit in ("relative", "   ") else tmp_path / explicit if explicit else None)
    assert Path(client.context()["dsh_home"]) == chosen
    assert client._environment()["DSH_HOME"] == str(chosen)
    assert client._environment()["REFACTORY_MODEL_DSH_HOME"] == str(chosen)
    seen = {}

    def capture(command, **kwargs):
        seen.update(kwargs["env"])
        return _FakeProcess(0)

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", capture)
    # Containment rewrites HOME after the client resolved its original home.
    monkeypatch.setenv("HOME", str(tmp_path / "contained"))
    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    result = run_on_dsh(state, tmp_path / "run", model=client)
    assert result.returncode == 0, result.stderr
    assert seen["DSH_HOME"] == seen["REFACTORY_MODEL_DSH_HOME"] == str(chosen)


def test_runtime_refuses_unready_model_home_even_when_session_is_ready(tmp_path, monkeypatch):
    from refactory_dsh.model import DshModelClient

    session = tmp_path / "session"
    worker = tmp_path / "worker home"
    _profile(session, "graph", [dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(session))
    client = DshModelClient(dsh_home=worker)

    def forbidden(*args, **kwargs):
        pytest.fail("must inspect the unready model home before spawning")

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", forbidden)
    result = run_on_dsh({}, tmp_path / "run", model=client)
    assert result.returncode == 2
    assert str(worker / "profiles/graph") in result.stderr
    assert f"DSH_HOME='{worker}'" in result.stderr
    assert dsh_profile.requirement() is None


def test_runtime_without_client_uses_model_home(tmp_path, monkeypatch):
    worker = tmp_path / "worker"
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "session"))
    monkeypatch.setenv("REFACTORY_MODEL_DSH_HOME", str(worker))
    _profile(worker, "graph", [dsh_profile.GRAPH_BUNDLE])
    seen = {}

    def capture(command, **kwargs):
        seen.update(kwargs["env"])
        return _FakeProcess(0)

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", capture)
    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    assert run_on_dsh(state, tmp_path / "run").returncode == 0
    assert seen["DSH_HOME"] == seen["REFACTORY_MODEL_DSH_HOME"] == str(worker)
    assert dsh_profile.profile_dir() == tmp_path / "session/profiles/graph"
    assert any(step.profile == "graph" and step.kind == "add" for step in dsh_profile.provision_plan())


def test_doctor_reports_both_homes_and_checks_model_profile(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from refactory_dsh.cli import cmd_doctor

    session, worker = tmp_path / "session", tmp_path / "worker"
    for entry in dsh_profile.TARGETS:
        if entry.owned:
            _profile(session, entry.profile, [dsh_profile.LAYERS[key].bundle for key in entry.layers])
    monkeypatch.setenv("DSH_HOME", str(session))
    monkeypatch.setenv("REFACTORY_MODEL_DSH_HOME", str(worker))
    monkeypatch.setattr("refactory_dsh.sandbox.model_readiness", lambda *a, **kw: {
        "provider": "test", "model": "test", "status": "configured",
    })
    cmd_doctor(Namespace(provider="", model=""))
    output = capsys.readouterr().out
    assert "DSH installation/session home" in output and str(session) in output
    assert "DSH model home" in output and str(worker) in output
    assert "MISS" in next(line for line in output.splitlines() if "DSH model profile 'graph'" in line)
    assert str(worker / "profiles/graph") in output


@pytest.mark.parametrize("explicit", [False, True])
def test_runtime_real_child_receives_preflight_home(tmp_path, monkeypatch, explicit):
    """A harmless launcher records the actual child environment; no model call."""
    from refactory_dsh.model import DshModelClient

    selected = tmp_path / "selected"
    _profile(selected, "graph", [dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "empty-session"))
    monkeypatch.setenv("REFACTORY_MODEL_DSH_HOME", str(tmp_path / "empty-worker" if explicit else selected))
    record = tmp_path / "received.json"
    launcher = tmp_path / "record_home.py"
    launcher.write_text(
        f"#!{sys.executable}\nimport json, os\nfrom pathlib import Path\n"
        f"Path({str(record)!r}).write_text(json.dumps({{key: os.environ[key] for key in "
        "['HOME', 'DSH_HOME', 'REFACTORY_MODEL_DSH_HOME']}))\n"
    )
    launcher.chmod(0o700)
    client = DshModelClient(dsh_home=selected if explicit else None, dsh_cli=str(launcher))
    monkeypatch.setenv("HOME", str(tmp_path / "contained"))
    state = registry.resolve(REPO_ROOT, "circle_packing/adaptive")[1]().state_dict()
    result = run_on_dsh(state, tmp_path / "run", model=client)
    assert result.returncode == 0, result.stderr
    received = json.loads(record.read_text())
    assert received["HOME"] == str(tmp_path / "contained")
    assert received["DSH_HOME"] == received["REFACTORY_MODEL_DSH_HOME"] == str(selected)


def test_a_profile_linked_to_another_checkout_is_a_mismatch(tmp_path, monkeypatch):
    """A profile whose symlink resolves to another checkout is not ready."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    other_checkout = tmp_path / "other-checkout" / "packages" / "bundle" / "graph"
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, other_checkout)
    monkeypatch.setenv("DSH_HOME", str(home))

    problem = dsh_profile.requirement()
    assert problem is not None
    assert str(other_checkout) in problem or "other-checkout" in problem
    assert dsh_profile.add_command() in problem
    assert not dsh_profile.is_ready()


def test_a_profile_linked_to_this_checkout_is_still_ready(tmp_path, monkeypatch):
    """A profile whose symlink resolves to this checkout is ready."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE,
                 REPO_ROOT / "packages" / "bundle" / "graph")
    monkeypatch.setenv("DSH_HOME", str(home))

    assert dsh_profile.is_ready()
    assert dsh_profile.requirement() is None
    dsh_profile.require()  # must not raise


def test_a_profile_without_a_symlink_falls_through(tmp_path, monkeypatch):
    """No symlink means no mismatch — the existing checks still govern."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    # No _link_bundle call — no node_modules symlink exists.
    monkeypatch.setenv("DSH_HOME", str(home))

    # The bundle is declared, no symlink to compare → still ready.
    assert dsh_profile.requirement() is None
    assert dsh_profile.is_ready()


def test_run_on_dsh_refuses_before_spawning_when_the_profile_links_to_another_checkout(
    tmp_path, monkeypatch
):
    """The runner refuses before spawning when the profile links to another checkout."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    other = tmp_path / "other" / "packages" / "bundle" / "graph"
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, other)
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("REFACTORY_MODEL_DSH_HOME", raising=False)

    def _no_spawn(*args, **kwargs):
        raise AssertionError("a subprocess was spawned without a usable profile")

    monkeypatch.setattr("refactory_dsh.dsh_runtime.subprocess.Popen", _no_spawn)
    run = run_on_dsh({}, tmp_path / "run")
    assert run.returncode == 2
    assert "other" in run.stderr  # names the other checkout
    assert dsh_profile.add_command() in run.stderr


def test_readiness_gaps_includes_checkout_mismatch(tmp_path, monkeypatch):
    """Doctor's readiness_gaps reports a checkout mismatch for an owned target."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    other = tmp_path / "other" / "packages" / "bundle" / "graph"
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, other)
    # Also create a ready GUI profile so only graph is a gap.
    _profile(home, dsh_profile.GUI_PROFILE, [
        "@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app",
        dsh_profile.PRESETS_BUNDLE, dsh_profile.WEB_SURFACE_BUNDLE,
    ])
    monkeypatch.setenv("DSH_HOME", str(home))

    gaps = dsh_profile.readiness_gaps()
    graph_gaps = [(p, b, r) for p, b, r in gaps if p == "graph"]
    assert len(graph_gaps) == 1
    assert "other" in graph_gaps[0][2]  # names the other checkout
    assert dsh_profile.add_command() in graph_gaps[0][2]


def test_bundle_checkout_root_returns_resolved_path(tmp_path, monkeypatch):
    """bundle_checkout_root resolves the profile's bundle symlink."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    target = tmp_path / "checkout" / "packages" / "bundle" / "graph"
    _link_bundle(home, "graph", dsh_profile.GRAPH_BUNDLE, target)
    monkeypatch.setenv("DSH_HOME", str(home))

    result = dsh_profile.bundle_checkout_root()
    assert result is not None
    assert result == target.resolve()


def test_bundle_checkout_root_returns_none_without_symlink(tmp_path, monkeypatch):
    """bundle_checkout_root returns None when no symlink exists."""
    home = tmp_path / "home"
    _profile(home, "graph", ["@deepseek-ai/dsh-base", dsh_profile.GRAPH_BUNDLE])
    monkeypatch.setenv("DSH_HOME", str(home))

    assert dsh_profile.bundle_checkout_root() is None


def test_additive_profile_is_composed_and_not_retired(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("DSH_HOME", str(home))
    assert [tuple(step) for step in dsh_profile.provision_plan(additive=True)] == [
        ("add", "graph", "graph"),
        ("init", "refactory-web", "web"),
        ("add", "refactory-web", "refactory"),
        ("add", "refactory-web", "refactory-web"),
    ]
    _ready(home)
    directory = _profile(home, "refactory-web", [
        "@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app",
        dsh_profile.PRESETS_BUNDLE, dsh_profile.WEB_SURFACE_BUNDLE,
    ])
    _declare(directory, [dsh_profile.PRESETS_BUNDLE, dsh_profile.WEB_SURFACE_BUNDLE])
    manifest_path = directory / "package.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dsh"]["profile"]["refactoryAdditive"] = True
    manifest_path.write_text(json.dumps(manifest))
    assert dsh_profile.provision_plan(additive=True) == []
    assert dsh_profile.provision_plan() == [], "standalone launch must not retire the additive profile"
    assert dsh_profile.obsolete_selections("refactory-web") == []


def _manifest(home: Path, profile: str, bundles: list[str], *, checkout: Path) -> None:
    """A profile manifest shaped like the harness writes: declarations and the link."""
    (home / "profiles" / profile).mkdir(parents=True, exist_ok=True)
    (home / "profiles" / profile / "package.json").write_text(json.dumps({
        "name": f"dsh-profile-{profile}",
        "dependencies": {bundle: f"link:{checkout}/packages/bundle/{bundle.split('/')[-1].removeprefix('dsh-')}"
                         for bundle in bundles},
        "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base", *bundles]}},
    }))


def _linked(home: Path, profile: str, bundle: str, checkout: Path) -> None:
    link = home / "profiles" / profile / "node_modules" / "@deepseek-ai" / bundle.split("/")[-1]
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(checkout / "packages/bundle" / bundle.split("/")[-1].removeprefix("dsh-"))


def _checkouts(tmp_path: Path) -> tuple[Path, Path, Path]:
    home, first, second = tmp_path / "dsh", tmp_path / "first", tmp_path / "second"
    for checkout in (first, second):
        (checkout / "packages/bundle/graph").mkdir(parents=True)
    return home, first, second


def test_the_plan_repoints_a_layer_that_resolves_into_another_checkout(tmp_path, monkeypatch):
    """Installing a checkout makes it the active one: a layer the profile already
    declares, but whose link points elsewhere, is a reason to add this checkout's copy."""
    from refactory_dsh import dsh_profile

    home, first, second = _checkouts(tmp_path)
    monkeypatch.setenv("DSH_HOME", str(home))
    _manifest(home, "graph", ["@deepseek-ai/dsh-graph"], checkout=first)
    _linked(home, "graph", "@deepseek-ai/dsh-graph", first)
    monkeypatch.setattr(dsh_profile, "REPO_ROOT", second)

    assert "add\tgraph\tgraph\n" in dsh_profile.render_plan(), (
        "a layer pointing at another checkout is added again")
    assert dsh_profile.resolves_elsewhere("graph", "@deepseek-ai/dsh-graph")

    # `dsh plugin add` moves the link; the plan then stops asking for it.
    _manifest(home, "graph", ["@deepseek-ai/dsh-graph"], checkout=second)
    _linked(home, "graph", "@deepseek-ai/dsh-graph", second)
    assert "add\tgraph\tgraph\n" not in dsh_profile.render_plan()
    assert not dsh_profile.resolves_elsewhere("graph", "@deepseek-ai/dsh-graph")


def test_a_second_checkout_takes_the_profile_over_and_the_first_reports_the_mismatch(tmp_path, monkeypatch):
    """What an install does now: the plan asks for the row, the link moves to the second
    checkout, and the first checkout's doctor reports the mismatch instead of a later
    command crashing."""
    from refactory_dsh import dsh_profile

    home, first, second = _checkouts(tmp_path)
    monkeypatch.setenv("DSH_HOME", str(home))
    _manifest(home, "graph", ["@deepseek-ai/dsh-graph"], checkout=first)
    _linked(home, "graph", "@deepseek-ai/dsh-graph", first)

    monkeypatch.setattr(dsh_profile, "REPO_ROOT", second)
    assert "add\tgraph\tgraph\n" in dsh_profile.render_plan()
    _manifest(home, "graph", ["@deepseek-ai/dsh-graph"], checkout=second)
    _linked(home, "graph", "@deepseek-ai/dsh-graph", second)
    assert dsh_profile.bundle_checkout_root("graph") == second / "packages/bundle/graph"

    monkeypatch.setattr(dsh_profile, "REPO_ROOT", first)
    said = dsh_profile.checkout_mismatch("graph")
    assert said is not None and str(second) in said, (
        "the first checkout reports where the profile points now")
