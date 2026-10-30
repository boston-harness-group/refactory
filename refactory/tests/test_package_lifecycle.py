"""The package lifecycle: two roots, provenance and trust, add/import/remove/verify.

The lifecycle issue: discovery and grouping were implemented and the verbs were not,
and generated packages were written into the source tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import registry
from refactory_dsh.registry import PackageClash, PackageError

MODE = "circle_packing/adaptive"


def _make_package(root: Path, name: str, *, requires: tuple[str, ...] = ()) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    requirement = f"requires={requires!r}, " if requires else ""
    (root / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n\n"
        f"PACKAGE = Package(name={name!r}, description='a test package', {requirement})\n"
    )
    return root


# ── two roots ──────────────────────────────────────────────────────────────
def test_the_workspace_is_not_the_install_tree(tmp_path):
    assert registry.workspace_root(tmp_path) != registry.packages_root()
    assert registry.workspace_root(tmp_path).is_relative_to(tmp_path)


def test_a_workspace_package_is_discovered_alongside_the_installed_ones(tmp_path):
    _make_package(registry.workspace_root(tmp_path) / "local_pkg", "local_pkg")
    names = registry.discover_names(tmp_path)
    assert "local_pkg" in names and "circle_packing" in names
    assert registry.where("local_pkg", tmp_path) == registry.workspace_root(tmp_path) / "local_pkg"


def test_a_name_in_both_roots_is_refused_rather_than_resolved(tmp_path):
    """Silent precedence is how a package gets shadowed and nobody finds out."""
    _make_package(registry.workspace_root(tmp_path) / "circle_packing", "circle_packing")
    with pytest.raises(PackageClash, match="both the install and"):
        registry.discover_names(tmp_path)


def test_a_workspace_package_is_importable_by_its_normal_path(tmp_path, monkeypatch):
    _make_package(registry.workspace_root(tmp_path) / "local_pkg", "local_pkg")
    registry.install_workspace(tmp_path)
    package = registry.load("local_pkg", tmp_path)
    assert package.name == "local_pkg"


@pytest.mark.parametrize("previous_valid", [False, True])
def test_loading_a_workspace_package_uses_the_requested_project(tmp_path, monkeypatch, previous_valid):
    import refactory_packages

    monkeypatch.setattr(refactory_packages, "__path__", list(refactory_packages.__path__))
    name = "project_scoped_pkg"
    previous, current = tmp_path / "previous", tmp_path / "current"
    old = _make_package(registry.workspace_root(previous) / name, name)
    if previous_valid:
        (old / "__init__.py").write_text(
            f"from refactory_dsh.nn import Package\nPACKAGE = Package(name={name!r}, description='previous')\n"
        )
        assert registry.load(name, previous).description == "previous"
    else:
        (old / "__init__.py").write_text("x = 1\n")
        registry.install_workspace(previous)
    _make_package(registry.workspace_root(current) / name, name)
    assert registry.load(name, current).description == "a test package"
    if previous_valid:
        # Switching back must also move a workspace already on the import path.
        assert registry.load(name, previous).description == "previous"
        assert registry.load(name, current).description == "a test package"


def test_the_catalogue_says_where_a_package_lives(tmp_path):
    _make_package(registry.workspace_root(tmp_path) / "local_pkg", "local_pkg")
    assert registry.where("local_pkg", tmp_path) == registry.workspace_root(tmp_path) / "local_pkg"
    grouped = registry.catalogue(tmp_path)
    assert grouped["local_pkg"] == [], "the test package declares no modes"
    assert grouped["circle_packing"][0]["where"] == str(
        registry.packages_root() / "circle_packing"
    )


# ── add and provenance ─────────────────────────────────────────────────────
def test_add_copies_a_package_into_the_workspace(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "added_pkg", "added_pkg")
    assert registry.add(tmp_path, source) == "added_pkg"
    assert (registry.workspace_root(tmp_path) / "added_pkg" / "__init__.py").exists()
    assert "added_pkg" in registry.discover_names(tmp_path)


def test_add_records_provenance_and_a_trust_tier(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "added_pkg", "added_pkg")
    registry.add(tmp_path, source)
    recorded = registry.provenance("added_pkg", tmp_path)
    assert recorded["trust"] == "generated"
    assert recorded["origin"] == str(source.resolve())
    assert recorded["at"], "provenance without a date is not provenance"
    assert registry.load("added_pkg", tmp_path).trust == "generated", (
        "the writer's record must outrank what the manifest claims about itself"
    )


def test_an_installed_package_is_trusted(tmp_path):
    assert registry.load("circle_packing", tmp_path).trust == "trusted"
    assert registry.provenance("circle_packing", tmp_path)["origin"] == "installed"


def test_add_refuses_a_name_clash(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "circle_packing", "circle_packing")
    with pytest.raises(PackageClash, match="already exists"):
        registry.add(tmp_path, source)


def test_add_can_rename_on_the_way_in(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "whatever", "renamed_pkg")
    assert registry.add(tmp_path, source, as_name="renamed_pkg") == "renamed_pkg"
    assert "renamed_pkg" in registry.discover_names(tmp_path)


def test_add_refuses_something_that_is_not_a_package(tmp_path):
    empty = tmp_path / "not_a_package"
    empty.mkdir()
    with pytest.raises(PackageError, match="not a package"):
        registry.add(tmp_path, empty)


def test_a_refused_add_leaves_no_residue(tmp_path):
    """A refusal after the copy must undo it, or the next add is a false clash.

    ``add`` is allowed to rename on the way in, so a name disagreement is no longer
    the refusal; a package that copies but cannot be loaded still is. The copy
    happens before ``load`` checks the package, so the failure has to evict it --
    the residue would make the next attempt fail with a false name clash that tells
    the caller to pass ``--as``, which is what they already did.
    """
    source = _make_package(tmp_path / "elsewhere" / "broken_pkg", "broken_pkg")
    (source / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n"
        "PACKAGE = Package(name='broken_pkg', description='a test package')\n"
        "raise RuntimeError('this package cannot load')\n"
    )
    destination = registry.workspace_root(tmp_path) / "broken_pkg"
    with pytest.raises(RuntimeError, match="cannot load"):
        registry.add(tmp_path, source)
    assert not destination.exists(), (
        "a refused add left the copied package behind, so the next attempt is a "
        "false name clash"
    )


def test_import_needs_a_path_or_a_url(tmp_path):
    with pytest.raises(PackageError, match="neither a path"):
        registry.import_package(tmp_path, "some/where/that/does/not/exist")


def test_import_from_a_path_is_add(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "imported_pkg", "imported_pkg")
    assert registry.import_package(tmp_path, str(source)) == "imported_pkg"


# ── remove ─────────────────────────────────────────────────────────────────
def test_remove_deletes_a_workspace_package(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "temp_pkg", "temp_pkg")
    registry.add(tmp_path, source)
    registry.remove(tmp_path, "temp_pkg")
    assert "temp_pkg" not in registry.discover_names(tmp_path)


def test_remove_refuses_an_installed_package(tmp_path):
    with pytest.raises(PackageError, match="read-only"):
        registry.remove(tmp_path, "circle_packing")


def test_the_shipped_example_declares_its_dependency(tmp_path):
    """The real case: annealing_pack imports circle_packing.ops for RADII."""
    assert "circle_packing" in registry.load("annealing_pack", tmp_path).requires


def test_remove_refuses_to_break_a_dependent(tmp_path):
    base = _make_package(tmp_path / "elsewhere" / "base_pkg", "base_pkg")
    registry.add(tmp_path, base)
    dependent = _make_package(
        tmp_path / "elsewhere" / "dependent_pkg", "dependent_pkg", requires=("base_pkg",)
    )
    registry.add(tmp_path, dependent)
    with pytest.raises(PackageError, match="require 'base_pkg'"):
        registry.remove(tmp_path, "base_pkg")


def test_force_removes_a_dependency_anyway(tmp_path):
    base = _make_package(tmp_path / "elsewhere" / "base_pkg", "base_pkg")
    registry.add(tmp_path, base)
    dependent = _make_package(
        tmp_path / "elsewhere" / "dependent_pkg", "dependent_pkg", requires=("base_pkg",)
    )
    registry.add(tmp_path, dependent)
    registry.remove(tmp_path, "base_pkg", force=True)
    assert "base_pkg" not in registry.discover_names(tmp_path)


def test_verify_reports_a_missing_requirement(tmp_path):
    _make_package(
        registry.workspace_root(tmp_path) / "needy_pkg", "needy_pkg", requires=("nowhere",)
    )
    registry.install_workspace(tmp_path)
    assert "not installed" in registry.verify(tmp_path, "needy_pkg")["needy_pkg"][0]


def test_verify_reports_a_broken_mode(tmp_path):
    package = registry.workspace_root(tmp_path) / "broken"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from refactory_dsh import nn\n"
        "from refactory_dsh.nn import Package\n\n"
        "class Broken(nn.Mode):\n"
        "    name = 'broken'\n"
        "    def build(self):\n"
        "        g = nn.Graph()\n"
        "        g.add(nn.Node('only', 'op', call='no.such.module:fn'))\n"
        "        return g\n\n"
        "PACKAGE = Package(name='broken', modes=(Broken,))\n"
    )
    registry.install_workspace(tmp_path)
    findings = registry.verify(tmp_path, "broken")
    assert findings["broken"], "a mode whose call does not resolve must be reported"
    assert "no.such.module" in findings["broken"][0]


def test_verify_is_clean_for_every_installed_package(tmp_path):
    findings = registry.verify(tmp_path)
    for package_name, problems in findings.items():
        assert problems == [], f"{package_name}: {problems}"


# ── the command line ───────────────────────────────────────────────────────
def test_the_cli_verify_verb(tmp_path, capsys):
    assert registry.main(["verify", "--project", str(tmp_path)]) == 0
    assert "circle_packing" in capsys.readouterr().out


def test_the_cli_add_and_remove_verbs(tmp_path, capsys):
    source = _make_package(tmp_path / "elsewhere" / "cli_pkg", "cli_pkg")
    assert registry.main(["add", str(source), "--project", str(tmp_path)]) == 0
    assert "cli_pkg" in capsys.readouterr().out
    assert registry.main(["remove", "cli_pkg", "--project", str(tmp_path)]) == 0
    assert "removed" in capsys.readouterr().out


def test_the_cli_reports_a_refusal_without_a_traceback(tmp_path, capsys):
    assert registry.main(["remove", "circle_packing", "--project", str(tmp_path)]) == 2
    assert "read-only" in capsys.readouterr().err


def test_the_cli_enable_and_disable(tmp_path, capsys):
    assert registry.main(["disable", "circle_packing", "--project", str(tmp_path)]) == 0
    assert registry.is_enabled(tmp_path, "circle_packing") is False
    assert registry.main(["enable", "circle_packing", "--project", str(tmp_path)]) == 0
    assert registry.is_enabled(tmp_path, "circle_packing") is True


def test_the_cli_smoke_verb_runs_a_mode(tmp_path, capsys):
    assert registry.main(["smoke", MODE, "--project", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_ok"] is True
    assert payload["score"] is not None


def test_the_listing_marks_a_package_that_is_not_trusted(tmp_path):
    source = _make_package(tmp_path / "elsewhere" / "sketchy", "sketchy")
    registry.add(tmp_path, source)
    listing = registry.render_catalogue(tmp_path)
    assert "sketchy/" in listing and "generated" in listing
