"""Packages: discovery, identity, and the invariant that every mode is runnable.

The last test here is the strongest statement the library can make about a
package it did not write: every mode of every installed package builds a graph
that validates with all of its calls resolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import nn, registry


def test_every_installed_package_mode_validates():
    """A mode package that cannot be validated must not be discoverable at all.

    Strategy-only packages (those with ``modes=()``) are valid: they contribute
    an outer-loop strategy rather than a mode, so there is nothing to validate
    here.  Any future strategy package (GEPA, etc.) falls into this category.
    """
    names = registry.discover_names()
    assert names, "no packages are installed"
    for name in names:
        package = registry.load(name)
        if not package.modes:
            continue  # strategy-only package — no modes to validate
        for mode_class in package.modes:
            identifier = f"{name}/{mode_class.name}"
            mode = mode_class()
            issues = nn.validate_state(
                mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
            )
            assert issues == [], f"{identifier}: {[str(i) for i in issues]}"
            # Scored, or deliberately unscored with a stated reason. Refusing the
            # second would rule out a mode whose subject is producing something
            # rather than maximising a number — the meta-modes — and accepting it
            # silently is how a mode that produces no signal goes unnoticed.
            assert mode.is_scored() or mode.unscored_reason, (
                f"{identifier} produces no signal and does not say why"
            )
            assert mode.parameters(), f"{identifier} exposes no knob"


def test_every_module_a_node_calls_is_one_its_package_names():
    """A resumed run pins the modules `ops` names (`recovery`), so one a node calls but
    `ops` leaves out could change under it. design's `delivery` was, until it was listed."""
    for name in registry.discover_names():
        package = registry.load(name)
        for mode_class in package.modes:
            for node in mode_class().graph().nodes.values():
                for called in filter(None, (node.call, node.evaluator)):
                    assert called.split(":")[0] in package.ops, (
                        f"{name}/{mode_class.name}: node {node.id!r} calls {called}, whose module "
                        "the package's ops do not name")


def test_a_package_cannot_be_named_outside_the_convention():
    for bad in ("CirclePacking", "1bad", "", "with-dash"):
        with pytest.raises(nn.PackageError):
            nn.mode_id(bad, "adaptive")


def test_a_bare_mode_name_is_refused():
    with pytest.raises(nn.PackageError, match="not package-qualified"):
        nn.split("adaptive")


def test_two_modes_in_one_package_cannot_share_a_name():
    class One(nn.Mode):
        name = "same"

        def build(self) -> nn.Graph:
            return nn.Graph()

    class Two(nn.Mode):
        name = "same"

        def build(self) -> nn.Graph:
            return nn.Graph()

    with pytest.raises(nn.PackageError, match="both named"):
        nn.Package(name="pkg", modes=(One, Two))


def test_the_catalogue_groups_modes_by_package(tmp_path):
    grouped = registry.catalogue(tmp_path)
    assert "circle_packing" in grouped
    for mode in grouped["circle_packing"]:
        assert mode["id"].startswith("circle_packing/")
        assert mode["package_enabled"] is True


def test_a_disabled_package_is_not_resolvable(tmp_path):
    identifier = "circle_packing/adaptive"
    assert registry.resolve(tmp_path, identifier)
    registry.set_enabled(tmp_path, "circle_packing", False)
    try:
        assert registry.is_enabled(tmp_path, "circle_packing") is False
        with pytest.raises(nn.PackageError, match="disabled"):
            registry.resolve(tmp_path, identifier)
        assert "circle_packing/adaptive" not in registry.enabled_packages(tmp_path)
    finally:
        registry.set_enabled(tmp_path, "circle_packing", True)
    assert registry.resolve(tmp_path, identifier)


def test_describing_a_mode_shows_its_surface():
    described = registry.describe_mode(Path("."), "circle_packing/adaptive")
    assert described["package"] == "circle_packing"
    assert described["scored"] is True
    assert set(described["knobs"]) >= {"grid", "order", "budget", "seed"}
    assert described["knobs"]["seed"]["frozen"] is True
    assert any(node["id"] == "budget_check" for node in described["nodes"])
    assert any(edge["condition"] == "reloop" for edge in described["edges"])


def test_an_unknown_mode_is_refused_with_the_package():
    with pytest.raises(nn.PackageError, match="has no mode"):
        registry.resolve(Path("."), "circle_packing/nonexistent")


def test_an_unknown_package_is_refused():
    with pytest.raises(nn.PackageError, match="cannot import package"):
        registry.load("no_such_package")


def test_the_listing_names_every_mode():
    listing = registry.render_catalogue(Path("."))
    for name in registry.discover_names():
        assert f"{name}/" in listing
        package = registry.load(name)
        for mode_class in package.modes:
            assert f"{name}/{mode_class.name}" in listing


# ── the entry points ───────────────────────────────────────────────────────
def test_the_list_verb_prints_every_package(capsys):
    assert registry.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "circle_packing/adaptive" in out


def test_the_list_verb_can_answer_in_json(capsys):
    import json

    assert registry.main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "circle_packing" in payload
    assert payload["circle_packing"][0]["id"]


def test_the_show_verb_describes_a_mode(capsys):
    assert registry.main(["show", "circle_packing/adaptive"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "circle_packing/adaptive"
    assert payload["package"] == "circle_packing"


def test_show_without_a_mode_is_an_error(capsys):
    assert registry.main(["show"]) == 2
    assert "needs a" in capsys.readouterr().err


def test_show_on_an_unknown_mode_reports_the_reason(capsys):
    assert registry.main(["show", "circle_packing/nope"]) == 2
    assert "has no mode" in capsys.readouterr().err


def test_the_loop_entry_point_reports_an_unusable_mode(capsys, tmp_path):
    from refactory_dsh import loop as loop_module

    code = loop_module.main([
        "no_such_package/mode", "--optimizer", "random",
        "--runs-root", str(tmp_path / "runs"),
    ])
    assert code == 2
    assert "cannot import package" in capsys.readouterr().err


def test_the_create_entry_point_needs_a_name():
    import pytest as _pytest

    from refactory_dsh import create as create_module

    with _pytest.raises(SystemExit):
        create_module.main(["--brief", "no name"])
