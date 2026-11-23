"""An edit keeps every file of the package it revises that its answer leaves alone.

`materialise` used to clear the staged package and write only the emitted files, and
`register` then swapped that in whole: an edit that rewrote `ops.py` deleted the
package's README, its data and anything else that is not Python. An edit now starts
from the package it revises; a `===DELETE: <path>===` block is how an answer removes a
file. A create (no package to start from) holds what it wrote, as before.

Each test drives the production steps on a real package in a project of its own:
`prepare_create` -> `materialise` -> `register`, with the registry reading the result in
the sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import create_ops, registry
from refactory_dsh.create_ops import (
    materialise, parse_emission, parse_files, prepare_create, register,
)
from tests_support import good_files, render

NAME = "keep_files_pkg"

README = "# keep_files_pkg\n\nCounts to a target. Thresholds live in `data/plate.csv`.\n"
PLATE = "well,value\nA1,0.25\nA2,0.75\n"
NOTES = "kept beside the code; not Python\n"


class Ctx:
    """The parts of a run context these steps use, writing into the run directory."""

    def __init__(self, workdir: Path, facts: dict[str, Any] | None = None):
        self.workdir = workdir
        self.facts: dict[str, Any] = dict(facts or {})
        self.knobs: dict[str, Any] = {}
        self.task: dict[str, Any] = {}
        self.model = None
        self.budget = None

    def knob(self, name: str, default: Any = None) -> Any:
        return self.knobs.get(name, default)

    def read_text(self, name: str) -> str:
        return (self.workdir / name).read_text()

    def write_text(self, name: str, content: str) -> None:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.workdir / name
        return json.loads(path.read_text()) if path.exists() else default

    def write_json(self, name: str, payload: Any) -> None:
        self.write_text(name, json.dumps(payload, indent=2, default=str))

    def set_fact(self, name: str, value: Any) -> None:
        self.facts[name] = value

    def exists(self, name: str) -> bool:
        return (self.workdir / name).exists()


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    monkeypatch.chdir(project)
    monkeypatch.setattr(create_ops, "prepare",
                        lambda ctx: {"name": NAME, "brief": "revise the counter"})
    return project


def _install_source(project: Path) -> Path:
    """The package an edit starts from: working code, a README, data and a text file."""
    root = registry.workspace_root(project) / NAME
    for relative, content in {**good_files(NAME), "README.md": README,
                              "data/plate.csv": PLATE, "notes.txt": NOTES}.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(content)
    return root


def _start(tmp_path: Path) -> Ctx:
    ctx = Ctx(tmp_path / "run")
    ctx.workdir.mkdir()
    ctx.facts.update(prepare_create(ctx))
    return ctx


def _emit(ctx: Ctx, answer: str) -> dict[str, Any]:
    ctx.write_text("emit.txt", answer)
    produced = materialise(ctx)
    ctx.facts.update(produced)
    return produced


NEW_OPS = good_files(NAME)["ops.py"].replace(
    '"""Operations for the counter mode."""', '"""Operations for the counter mode, revised."""'
)


def test_an_edit_that_emits_only_ops_keeps_the_readme_and_the_data(tmp_path, project):
    _install_source(project)
    ctx = _start(tmp_path)
    produced = _emit(ctx, render({"ops.py": NEW_OPS}))
    assert produced["files"] == ["ops.py"]
    register(ctx)

    shared = registry.workspace_root(project) / NAME
    assert (shared / "README.md").read_text() == README, "the edit deleted the README"
    assert (shared / "data" / "plate.csv").read_text() == PLATE, "the edit deleted the data"
    assert (shared / "notes.txt").read_text() == NOTES
    assert (shared / "mode.py").read_text() == good_files(NAME)["mode.py"]
    assert (shared / "ops.py").read_text() == NEW_OPS
    manifest = ctx.read_json("staging/manifest.json")
    assert manifest["kept"] == ["README.md", "__init__.py", "data/plate.csv", "mode.py",
                                "notes.txt"]
    assert manifest["deleted"] == []
    changes = {c["path"]: c["kind"] for c in ctx.read_json("staging/package-diff.json")["changes"]}
    assert changes.get("ops.py") == "modified"
    assert "README.md" not in changes and "data/plate.csv" not in changes


def test_an_explicit_delete_removes_its_file_and_keeps_the_rest(tmp_path, project):
    _install_source(project)
    ctx = _start(tmp_path)
    _emit(ctx, render({"ops.py": NEW_OPS}) + "===DELETE: data/plate.csv===\n")
    register(ctx)

    shared = registry.workspace_root(project) / NAME
    assert not (shared / "data" / "plate.csv").exists(), "the deletion did not happen"
    assert (shared / "README.md").read_text() == README
    assert (shared / "notes.txt").read_text() == NOTES
    assert ctx.read_json("staging/manifest.json")["deleted"] == ["data/plate.csv"]
    changes = {c["path"]: c["kind"] for c in ctx.read_json("staging/package-diff.json")["changes"]}
    assert changes["data/plate.csv"] == "deleted"


def test_a_second_attempt_starts_from_the_source_not_from_the_first(tmp_path, project):
    """A reloop to emit applies the new answer to the package the run started from: a
    file only the first attempt wrote is gone, a file it deleted is back."""
    _install_source(project)
    ctx = _start(tmp_path)
    _emit(ctx, render({"ops.py": NEW_OPS, "extra.py": "X = 1\n"}) + "===DELETE: notes.txt===\n")
    staged = Path(ctx.facts["package_root"])
    assert (staged / "extra.py").exists() and not (staged / "notes.txt").exists()

    _emit(ctx, render({"ops.py": NEW_OPS}))
    assert not (staged / "extra.py").exists(), "the first attempt's file survived"
    assert (staged / "notes.txt").read_text() == NOTES
    assert (staged / "README.md").read_text() == README


def test_a_create_holds_only_what_it_wrote(tmp_path, project):
    ctx = _start(tmp_path)
    _emit(ctx, render(good_files(NAME)))
    staged = Path(ctx.facts["package_root"])
    assert sorted(p.name for p in staged.iterdir() if p.name != "__pycache__") == [
        ".origin.json", "__init__.py", "mode.py", "ops.py"]
    _emit(ctx, render({k: v for k, v in good_files(NAME).items()}))
    assert ctx.read_json("staging/manifest.json")["kept"] == []


def test_a_deletion_of_a_file_the_package_lacks_goes_back_to_emit(tmp_path, project):
    _install_source(project)
    ctx = _start(tmp_path)
    produced = _emit(ctx, render({"ops.py": NEW_OPS}) + "===DELETE: data/missing.csv===\n")
    assert "data/missing.csv" in produced["emission_refused"]
    assert "the package does not have" in produced["emission_refused"]
    # Nothing was applied: the staged package is still the source.
    staged = Path(ctx.facts["package_root"])
    assert (staged / "ops.py").read_text() == good_files(NAME)["ops.py"]


def test_the_report_says_when_the_package_has_no_readme(tmp_path, project):
    """C13: every package is meant to carry a README. The run's report says when the
    package it leaves has none, and writes none itself."""
    root = _install_source(project)
    ctx = Ctx(tmp_path / "run", {"name": NAME, "package_path": str(root)})
    assert create_ops._readme_report(ctx) == {"readme": "README.md"}
    (root / "README.md").unlink()
    reported = create_ops._readme_report(ctx)
    assert reported["readme"] is None
    assert "has no README" in reported["warnings"][0]
    assert not (root / "README.md").exists(), "the report must not fabricate a README"


# ── the emission format ───────────────────────────────────────────────────


def test_parse_emission_reads_files_and_deletions():
    files, deletes = parse_emission(
        "===FILE: ops.py===\nx = 1\n===DELETE: data/a.csv===\n===DELETE: b.txt===\n"
    )
    assert files == {"ops.py": "x = 1\n"}
    assert deletes == ["data/a.csv", "b.txt"]
    assert parse_files("===FILE: ops.py===\nx = 1\n===DELETE: b.txt===\n") == {"ops.py": "x = 1\n"}


@pytest.mark.parametrize("answer, said", [
    ("===DELETE: a.py===\nsome text\n", "has text after it"),
    ("===FILE: a.py===\nx = 1\n===DELETE: a.py===\n", "both writes and deletes"),
    ("===DELETE: `a.py`===\n", "malformed emission header"),
])
def test_parse_emission_refuses_a_malformed_deletion(answer, said):
    with pytest.raises(ValueError, match=said):
        parse_emission(answer)


def test_a_second_prepare_starts_from_the_source_as_it_is_now(tmp_path, project):
    """A run that prepares again rebuilds its baseline: a file the source lost in between
    is not brought back from the first prepare's copy."""
    shared = _install_source(project)
    ctx = _start(tmp_path)
    (shared / "notes.txt").unlink()
    ctx.facts.update(prepare_create(ctx))
    _emit(ctx, render({"ops.py": NEW_OPS}))
    staged = Path(ctx.facts["package_root"])
    assert not (staged / "notes.txt").exists(), "the stale baseline survived a new prepare"
    assert (staged / "README.md").read_text() == README


def test_links_out_of_the_package_are_dropped_and_links_inside_stay_links(tmp_path, project):
    """A symlink is never copied as the file it points at. One that leaves the package
    (absolute or escaping) is dropped, as it was when an edit kept only what it wrote;
    one inside the package stays the same link through materialise and register."""
    import os

    shared = _install_source(project)
    outside = tmp_path / "secret.txt"
    outside.write_text("not part of the package\n")
    os.symlink(outside, shared / "absolute_out.txt")
    os.symlink(os.path.relpath(outside, shared / "data"), shared / "data" / "escaping.txt")
    os.symlink("plate.csv", shared / "data" / "latest.csv")
    ctx = _start(tmp_path)
    _emit(ctx, render({"ops.py": NEW_OPS}))
    staged = Path(ctx.facts["package_root"])
    for root in (staged, create_ops._baseline_root(staged)):
        assert not os.path.lexists(root / "absolute_out.txt")
        assert not os.path.lexists(root / "data" / "escaping.txt")
    register(ctx)

    for root in (staged, shared):
        assert not os.path.lexists(root / "absolute_out.txt"), "a link out was copied"
        assert not os.path.lexists(root / "data" / "escaping.txt"), "a link out was copied"
        assert (root / "data" / "latest.csv").is_symlink()
        assert os.readlink(root / "data" / "latest.csv") == "plate.csv"
        assert (root / "data" / "latest.csv").read_text() == PLATE


def test_a_write_in_place_clears_what_an_earlier_answer_left(tmp_path, project):
    """Without `prepare_create` there is no source package: the directory holds what the
    answer writes, and a file an earlier answer left there is gone."""
    root = registry.workspace_root(project) / NAME
    root.mkdir(parents=True)
    (root / "stale.py").write_text("LEFT = True\n")
    ctx = Ctx(tmp_path / "run", {"name": NAME})
    ctx.workdir.mkdir()
    produced = _emit(ctx, render(good_files(NAME)))
    assert Path(produced["package_path"]) == root
    assert not (root / "stale.py").exists(), "an earlier answer's file survived"
    assert (root / "ops.py").read_text() == good_files(NAME)["ops.py"]
    assert ctx.read_json("staging/manifest.json")["kept"] == []
