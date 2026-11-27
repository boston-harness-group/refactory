"""Declared-inputs inference: a created mode's inputs are inferred from its structure.

When ``--input`` is not given, ``empty_input.infer_inputs`` reads the produced package's
knobs and infers which are inputs:

* a ``required=True`` str knob explicitly needs user input;
* a knob whose sample or default is an existing path that looks like input data
  (under the project or home, not a system location, write or scratch);
* data-node ``source_path`` sources are handled separately by ``plan()``.

Outputs (knobs whose value names a declared write), configuration files (not required,
no input-like path), and non-str settings are never inferred.

The inferred inputs are stored in ``brief.json``'s ``requires_input``, so the empty-input
check and every other reader works without ``--input``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import empty_input, registry
from refactory_dsh.empty_input import infer_inputs

NAME = "infer_pkg"


# ── fixture modes ─────────────────────────────────────────────────────────

_INIT = '''\
"""A generated package for declared-inputs inference tests."""
from refactory_dsh.nn import Package

from .mode import {mode_class}

PACKAGE = Package(name="{name}", description="{description}", modes=({mode_class},),
                  ops=("refactory_packages.{name}.ops",))
'''

#: A mode with a required input knob: the mode stops on empty input because the
#: knob is required and the ops check for it.
_REQUIRED_MODE = '''\
"""A mode that declares its input as required."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Analyser(nn.Mode):
    name = "analyser"
    description = "analyse CSV files from a required input folder"

    @classmethod
    def declare(cls):
        return {{"input_dir": nn.Knob("", "str", description="folder of plate CSVs",
                                      required=True),
                 "threshold": nn.Knob(0.5, "float", lo=0.0, hi=1.0,
                                      description="pass/fail threshold")}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("analyse", "op", call=OPS + ":analyse",
                          knobs=("input_dir", "threshold"),
                          writes=("report.json",), produces=("rows",)))
        return graph

    def score(self, run):
        return float(run.facts["rows"])
'''

_REQUIRED_OPS = '''\
from pathlib import Path


def analyse(ctx):
    folder = Path(str(ctx.knob("input_dir", "") or "."))
    tables = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    if not tables:
        raise ValueError(f"no input: input_dir {ctx.knob('input_dir', '')!r} holds no CSV files")
    rows = sum(max(0, len(p.read_text().splitlines()) - 1) for p in tables)
    ctx.write_json("report.json", {"rows": rows})
    return {"rows": rows}
'''


def required_files(name: str) -> dict[str, str]:
    return {
        "__init__.py": _INIT.format(name=name, mode_class="Analyser",
                                    description="analyse CSVs"),
        "mode.py": _REQUIRED_MODE.format(name=name),
        "ops.py": _REQUIRED_OPS,
    }


#: A mode with a required input, a config knob, and an output folder in writes.
_MIXED_MODE = '''\
"""A mode with input, config and output knobs."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Mixed(nn.Mode):
    name = "mixed"
    description = "process data with config and output"

    @classmethod
    def declare(cls):
        return {{"input_dir": nn.Knob("", "str", description="input data folder",
                                      required=True),
                 "config_file": nn.Knob("", "str",
                                        description="settings file"),
                 "out_dir": nn.Knob("output", "str",
                                    description="where results go")}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("work", "op", call=OPS + ":work",
                          knobs=("input_dir", "config_file", "out_dir"),
                          writes=("output", "done.json"), produces=("count",)))
        return graph

    def score(self, run):
        return float(run.facts.get("count", 0))
'''

_MIXED_OPS = '''\
from pathlib import Path


def work(ctx):
    folder = Path(str(ctx.knob("input_dir", "") or "."))
    items = list(folder.iterdir()) if folder.is_dir() else []
    if not items:
        raise ValueError("no input data")
    out = Path(str(ctx.knob("out_dir", "output")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.txt").write_text(str(len(items)))
    ctx.write_json("done.json", {"count": len(items)})
    return {"count": len(items)}
'''


def mixed_files(name: str) -> dict[str, str]:
    return {
        "__init__.py": _INIT.format(name=name, mode_class="Mixed",
                                    description="mixed knob types"),
        "mode.py": _MIXED_MODE.format(name=name),
        "ops.py": _MIXED_OPS,
    }


#: A pure generator: no input knobs (only a frozen setting).
_GENERATOR_MODE = '''\
"""A mode that generates output without input data."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Generator(nn.Mode):
    name = "gen"
    description = "generate a fixed report"

    @classmethod
    def declare(cls):
        return {{"seed": nn.Knob(42, "int", lo=0, hi=1000,
                                 description="random seed")}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("gen", "op", call=OPS + ":gen",
                          knobs=("seed",),
                          writes=("output.json",), produces=("value",)))
        return graph

    def score(self, run):
        return float(run.facts.get("value", 0))
'''

_GENERATOR_OPS = '''\
def gen(ctx):
    import random
    rng = random.Random(int(ctx.knob("seed", 42)))
    value = rng.random()
    ctx.write_json("output.json", {"value": value})
    return {"value": value}
'''


def generator_files(name: str) -> dict[str, str]:
    return {
        "__init__.py": _INIT.format(name=name, mode_class="Generator",
                                    description="generate without input"),
        "mode.py": _GENERATOR_MODE.format(name=name),
        "ops.py": _GENERATOR_OPS,
    }


# ── helpers ───────────────────────────────────────────────────────────────

@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    monkeypatch.chdir(project)
    return project


@pytest.fixture
def plates(tmp_path) -> Path:
    folder = tmp_path / "plates"
    folder.mkdir()
    (folder / "p1.csv").write_text("well,value\nA1,0.1\nA2,0.2\nA3,0.3\n")
    return folder


def _install(project: Path, name: str, files: dict[str, str]) -> None:
    root = registry.workspace_root(project) / name
    for relative, content in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(content)
    registry.install_workspace(project)


def _run(project: Path, tmp_path: Path, identifier: str, samples: dict[str, str],
         declared: tuple[str, ...] = ()) -> dict:
    return empty_input.run(identifier, tmp_path / "empty-run", project=project,
                           samples=samples, declared=declared, model=False, timeout_s=120)


# ── T1: required knob inferred as input, check runs ──────────────────────


def test_required_knob_is_inferred_as_input(tmp_path, project, plates):
    """A mode with a required knob and no --input: the inference detects the required
    knob as input, the check runs, and the mode stops on empty input → pass."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser", {"input_dir": str(plates)})
    assert receipt["inferred"] is True
    assert "input_dir" in receipt["declared"]
    assert receipt["status"] != "not_applicable", (
        "with a required knob the check must run, not say 'not applicable'")
    assert receipt["status"] == "pass", receipt


def test_required_knob_inferred_without_sample(tmp_path, project):
    """No sample at all: the required knob's empty default becomes the empty string.
    The mode's required-knob check refuses to start → non-zero exit → pass."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser", {})
    assert receipt["inferred"] is True
    assert "input_dir" in receipt["declared"]
    assert receipt["status"] == "pass", receipt


# ── T2: --input overrides inference ──────────────────────────────────────


def test_explicit_input_overrides_inference(tmp_path, project, plates):
    """When --input is given (declared is non-empty), inference does not run."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser",
                   {"input_dir": str(plates)}, declared=("input_dir",))
    assert receipt.get("inferred") is not True
    assert receipt["status"] == "pass", receipt


def test_explicit_input_different_from_inferred(tmp_path, project, plates):
    """--input names a different knob than what inference would find: the declared
    knob is used, not the inferred one."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser",
                   {"input_dir": str(plates)}, declared=("threshold",))
    assert receipt.get("inferred") is not True
    assert "threshold" in receipt["knobs"], "the explicitly declared knob is emptied"
    assert "input_dir" not in receipt["knobs"], (
        "the inferred knob must not be emptied when --input is explicit")


# ── T3: pure generator → not_applicable ──────────────────────────────────


def test_pure_generator_is_not_applicable(tmp_path, project):
    """A mode with no str knobs and no data-node sources: nothing to infer,
    'not applicable', no false declaration."""
    _install(project, "gen_pkg", generator_files("gen_pkg"))
    receipt = _run(project, tmp_path, "gen_pkg/gen", {})
    assert receipt["inferred"] is False
    assert receipt["declared"] == []
    assert receipt["status"] == "not_applicable"


# ── T4: output/config never declared as input ────────────────────────────


def test_output_knob_in_writes_not_inferred(tmp_path, project):
    """A knob whose value names a file the graph writes is output, not input."""
    _install(project, "mixed_pkg", mixed_files("mixed_pkg"))
    receipt = _run(project, tmp_path, "mixed_pkg/mixed", {})
    assert "out_dir" not in receipt["declared"], "output folder must not be inferred as input"
    assert "config_file" not in receipt["declared"], "config must not be inferred as input"
    assert "input_dir" in receipt["declared"], "required input should be inferred"


def test_config_knob_not_inferred(tmp_path, project):
    """A config knob (not required, no input-like default) is never inferred."""
    described = {
        "knobs": ["config_file", "reference_db", "threshold"],
        "values": {"config_file": "", "reference_db": "/usr/share/data",
                   "threshold": 0.5},
        "required": [],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, {}, project=tmp_path, run_root=tmp_path / "run")
    assert result == [], f"no knob should be inferred: {result}"


def test_numeric_bool_enum_knobs_never_inferred(tmp_path, project):
    """Non-str knobs (float, int, bool, enum) are settings, never inputs."""
    described = {
        "knobs": ["budget", "verbose", "seed"],
        "values": {"budget": 3, "verbose": True, "seed": 42},
        "required": [],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, {}, project=tmp_path, run_root=tmp_path / "run")
    assert result == []


# ── T4b: knob with input-like default path inferred ──────────────────────


def test_knob_with_input_like_default_is_inferred(tmp_path, project):
    """A knob whose default is an existing path under the project that looks like input."""
    (project / "data" / "input").mkdir(parents=True)
    described = {
        "knobs": ["source_dir"],
        "values": {"source_dir": "data/input"},
        "required": [],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, {}, project=project, run_root=project / "run")
    assert result == ["source_dir"]


def test_knob_with_input_like_sample_is_inferred(tmp_path, project):
    """A knob whose sample is an existing path under the project is inferred."""
    data = project / "data" / "plates"
    data.mkdir(parents=True)
    described = {
        "knobs": ["input_dir"],
        "values": {"input_dir": ""},
        "required": [],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, {"input_dir": str(data)}, project=project,
                          run_root=project / "run")
    assert result == ["input_dir"]


# ── T5: mutation checks ──────────────────────────────────────────────────


def test_mutation_removing_inference_makes_required_knob_not_applicable(
    tmp_path, project, plates, monkeypatch,
):
    """Undo: monkeypatch infer_inputs to return nothing. The check falls back to
    'not applicable' because declared is empty and no data-node sources exist."""
    _install(project, NAME, required_files(NAME))
    monkeypatch.setattr(empty_input, "infer_inputs", lambda *a, **kw: [])
    receipt = _run(project, tmp_path, f"{NAME}/analyser", {"input_dir": str(plates)})
    assert receipt["status"] == "not_applicable", (
        "with inference disabled, the check must fall back to not_applicable")


def test_mutation_removing_required_field_loses_inference(tmp_path, project, plates):
    """Undo: install a mode identical to required_files but without required=True.
    The inference finds nothing (no required knob, no input-like default), so the
    check is 'not applicable'."""
    files = required_files(NAME)
    files["mode.py"] = files["mode.py"].replace("required=True", "")
    _install(project, NAME, files)
    receipt = _run(project, tmp_path, f"{NAME}/analyser", {"input_dir": str(plates)})
    assert receipt["status"] == "not_applicable", (
        "without required=True the knob is not inferred, and plates is under "
        "tmp_path (not project), so _looks_like_input returns False")


# ── brief.json is updated with inferred inputs ──────────────────────────


def test_check_empty_input_updates_brief_json(tmp_path, project, plates):
    """check_empty_input stores inferred inputs in brief.json so later readers see them."""
    from refactory_dsh import create_ops

    _install(project, NAME, required_files(NAME))
    package = registry.workspace_root(project) / NAME

    class Ctx:
        def __init__(self):
            self.workdir = tmp_path / "run"
            self.workdir.mkdir()
            self.facts = {"name": NAME, "package_path": str(package)}
            self.knobs = {"sample_knobs": json.dumps([f"input_dir={plates}"]),
                          "smoke_timeout_s": 120}
            self.model = self.budget = None

        def knob(self, name, default=None):
            return self.knobs.get(name, default)

        def read_json(self, name, default=None):
            path = self.workdir / name
            return json.loads(path.read_text()) if path.exists() else default

        def write_json(self, name, payload):
            (self.workdir / name).write_text(json.dumps(payload, default=str))

        def exists(self, name):
            return (self.workdir / name).exists()

    ctx = Ctx()
    ctx.write_json("smoke.json", {"mode": f"{NAME}/analyser"})
    ctx.write_json("brief.json", {"name": NAME, "brief": "analyse plates",
                                  "requires_input": []})
    receipt = create_ops.check_empty_input(ctx)
    brief = ctx.read_json("brief.json", {})
    assert "input_dir" in brief.get("requires_input", []), (
        "inferred inputs must be stored in brief.json")
    assert receipt["status"] == "pass"


# ── infer_inputs unit tests ──────────────────────────────────────────────


def test_infer_inputs_required_not_in_writes(tmp_path):
    """A required knob whose value names a declared write is NOT inferred."""
    described = {
        "knobs": ["output_path"],
        "values": {"output_path": "results"},
        "required": ["output_path"],
        "data_nodes": [],
        "writes": ["results"],
    }
    result = infer_inputs(described, {}, project=tmp_path, run_root=tmp_path / "run")
    assert result == []


def test_infer_inputs_multiple_required(tmp_path):
    """Multiple required knobs are all inferred."""
    described = {
        "knobs": ["input_dir", "query_text"],
        "values": {"input_dir": "", "query_text": ""},
        "required": ["input_dir", "query_text"],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, {}, project=tmp_path, run_root=tmp_path / "run")
    assert result == ["input_dir", "query_text"]
