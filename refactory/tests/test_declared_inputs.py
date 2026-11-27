"""Declared-inputs: inputs come from ``--input``, the plan, or a narrowed structural fallback.

The planning step (strategy) names the inputs the mode will take, which the person
reviews at the gate.  ``--input`` wins when given.  When neither ``--input`` nor the
plan declared inputs, ``infer_inputs`` applies a narrowed structural fallback: only
``required=True`` knobs whose value *and* sample are both empty or are an existing
input-like path.  A required setting with a non-empty non-path default (``model_name``,
``api_endpoint``) is never inferred.  A knob whose value only looks like a path stays
in ``plan()``'s ``undeclared`` list (inconclusive, never emptied).
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
    assert "input_dir" in receipt["declared"]
    assert receipt["status"] != "not_applicable", (
        "with a required knob the check must run, not say 'not applicable'")
    assert receipt["status"] == "pass", receipt


def test_required_knob_inferred_without_sample(tmp_path, project):
    """No sample at all: the required knob's empty default becomes the empty string.
    The mode's required-knob check refuses to start → non-zero exit → pass."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser", {})
    assert "input_dir" in receipt["declared"]
    assert receipt["status"] == "pass", receipt


# ── T2: --input overrides inference ──────────────────────────────────────


def test_explicit_input_overrides_inference(tmp_path, project, plates):
    """When --input is given (declared is non-empty), inference does not run."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser",
                   {"input_dir": str(plates)}, declared=("input_dir",))
    assert receipt["status"] == "pass", receipt


def test_explicit_input_different_from_inferred(tmp_path, project, plates):
    """--input names a different knob than what inference would find: the declared
    knob is used, not the inferred one."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser",
                   {"input_dir": str(plates)}, declared=("threshold",))
    assert "threshold" in receipt["knobs"], "the explicitly declared knob is emptied"
    assert "input_dir" not in receipt["knobs"], (
        "the inferred knob must not be emptied when --input is explicit")


# ── T3: pure generator → not_applicable ──────────────────────────────────


def test_pure_generator_is_not_applicable(tmp_path, project):
    """A mode with no str knobs and no data-node sources: nothing to infer,
    'not applicable', no false declaration."""
    _install(project, "gen_pkg", generator_files("gen_pkg"))
    receipt = _run(project, tmp_path, "gen_pkg/gen", {})
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
    result = infer_inputs(described, project=tmp_path, run_root=tmp_path / "run")
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
    result = infer_inputs(described, project=tmp_path, run_root=tmp_path / "run")
    assert result == []


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
    result = infer_inputs(described, project=tmp_path, run_root=tmp_path / "run")
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
    result = infer_inputs(described, project=tmp_path, run_root=tmp_path / "run")
    assert result == ["input_dir", "query_text"]


# ── the seven required tests ────────────────────────────────────────────


def test_plan_declared_inputs_appear_in_gate_text_and_check_runs(
    tmp_path, project, plates,
):
    """With no --input, the plan (strategy.json) names the inputs. review_design
    extracts them into brief.json, the gate_strategy notes render them, and the
    empty-input check runs with those inputs declared."""
    from refactory_dsh import create_ops

    _install(project, NAME, required_files(NAME))
    strategy = json.dumps({
        "considered": [{"approach": "a", "why_not": ""}, {"approach": "b", "why_not": ""}],
        "chosen": "a", "why": "simpler", "will_not": [], "closest_existing": "none",
        "knobs": [{"name": "input_dir", "kind": "str", "default": "",
                   "description": "folder of CSVs", "required": True}],
        "nodes": [{"id": "analyse", "kind": "op", "call": "ops:analyse",
                   "knobs": ["input_dir"], "writes": ["report.json"], "produces": ["rows"]}],
        "edges": [], "start": "analyse", "score": "rows",
        "inputs": ["input_dir"],
        "findings": {},
    })
    # Simulate the review gate context.
    class Ctx:
        def __init__(self):
            self.workdir = tmp_path / "run"
            self.workdir.mkdir()
            self.facts = {"name": NAME, "requires_input": ""}
            self.knobs = {"design_attempts": 2}
        def knob(self, name, default=None):
            return self.knobs.get(name, default)
        def read_json(self, name, default=None):
            path = self.workdir / name
            return json.loads(path.read_text()) if path.exists() else default
        def read_text(self, name):
            return (self.workdir / name).read_text()
        def write_json(self, name, payload):
            (self.workdir / name).write_text(json.dumps(payload, default=str))
        def write_text(self, name, text):
            (self.workdir / name).write_text(text)
        def set_fact(self, name, value):
            self.facts[name] = value
        def exists(self, name):
            return (self.workdir / name).exists()

    ctx = Ctx()
    ctx.write_json("brief.json", {"name": NAME, "brief": "analyse plates",
                                  "requires_input": []})
    (ctx.workdir / "strategy.json").write_text(strategy)

    verdict = create_ops.review_design(ctx)
    assert verdict == "proceed"
    brief = ctx.read_json("brief.json", {})
    assert brief["requires_input"] == ["input_dir"], (
        "the plan's inputs must be written to brief.json")
    assert brief.get("requires_input_source") == "plan"
    assert "input_dir" in ctx.facts.get("requires_input", ""), (
        "the requires_input fact must name the declared input")
    # The gate_strategy notes render the fact.
    from refactory_dsh.nn.executor import render, RunContext
    from refactory_dsh.create import CreateMode
    gate = CreateMode().build().nodes["gate_strategy"]
    run_ctx = RunContext(ctx.workdir)
    run_ctx.facts = dict(ctx.facts)
    rendered = render(gate.notes, run_ctx)
    assert "input_dir" in rendered, (
        "the gate text must show the declared input so the person can correct it")
    # Now run the empty-input check with brief.json's declared inputs.
    receipt = _run(project, tmp_path, f"{NAME}/analyser",
                   {"input_dir": str(plates)},
                   declared=tuple(brief["requires_input"]))
    assert receipt["status"] == "pass", receipt


def test_input_overrides_plan(tmp_path, project, plates):
    """--input overrides the plan's inputs: even if the strategy names different
    inputs, the explicitly declared ones are used."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser",
                   {"input_dir": str(plates)}, declared=("input_dir",))
    assert receipt["status"] == "pass", receipt
    assert "input_dir" in receipt["knobs"], "the --input knob is emptied"


def test_pure_generator_not_applicable(tmp_path, project):
    """A mode with no input knobs and no data-node sources gets 'not applicable'."""
    _install(project, "gen_pkg", generator_files("gen_pkg"))
    receipt = _run(project, tmp_path, "gen_pkg/gen", {})
    assert receipt["declared"] == []
    assert receipt["status"] == "not_applicable"


def test_required_model_name_setting_not_inferred(tmp_path):
    """F1: a required knob with a non-empty non-path default is a setting, not an input.
    model_name='gpt-4' is required but has a fallback value, so it must not be inferred."""
    described = {
        "knobs": ["model_name", "api_endpoint", "input_dir"],
        "values": {"model_name": "gpt-4", "api_endpoint": "https://api.example.com",
                   "input_dir": ""},
        "required": ["model_name", "api_endpoint", "input_dir"],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, project=tmp_path, run_root=tmp_path / "run")
    assert "model_name" not in result, (
        "a required setting with a non-empty default must not be inferred")
    assert "api_endpoint" not in result, (
        "a required setting with a non-empty default must not be inferred")
    assert "input_dir" in result, (
        "a required knob with an empty default is input")


def test_config_file_stays_undeclared_inconclusive_not_emptied(tmp_path, project, plates):
    """F2: a config_file='config.yaml' knob whose value is an existing input-like path
    stays undeclared (inconclusive), and is never emptied."""
    config = project / "config.yaml"
    config.write_text("threshold: 0.5\n")
    described = {
        "knobs": ["config_file", "input_dir"],
        "values": {"config_file": "config.yaml", "input_dir": ""},
        "required": ["input_dir"],
        "data_nodes": [],
        "writes": [],
    }
    result = infer_inputs(described, project=project, run_root=project / "run")
    assert "config_file" not in result, (
        "a non-required knob must not be inferred even if its value is an input-like path")
    assert "input_dir" in result
    # plan() leaves it in undeclared → inconclusive
    planned = empty_input.plan(described, {}, ("input_dir",),
                               empty_root=tmp_path / "e", project=project,
                               run_root=project / "run")
    assert any("config_file" in entry for entry in planned["undeclared"]), (
        "config_file must stay in the undeclared list")
    assert "config_file" not in planned["knobs"], (
        "config_file must not be emptied")


def test_fallback_warning_appears(tmp_path, project, plates, capsys):
    """F3: when the structural fallback fires, a run warning names what was inferred."""
    _install(project, NAME, required_files(NAME))
    receipt = _run(project, tmp_path, f"{NAME}/analyser", {"input_dir": str(plates)})
    assert receipt["status"] == "pass", receipt
    captured = capsys.readouterr()
    assert "inferred" in captured.err, (
        "a run warning must name what the fallback inferred")
    assert "input_dir" in captured.err


# ── adopt_strategy_inputs: reloop replaces plan inputs ───────────────────


def _make_ctx(tmp_path, brief_payload=None):
    """Lightweight gate context for _adopt_strategy_inputs tests."""

    class Ctx:
        def __init__(self):
            self.workdir = tmp_path / "run"
            self.workdir.mkdir(exist_ok=True)
            self.facts = {"name": NAME, "requires_input": ""}
            self.knobs = {"design_attempts": 2}
        def knob(self, name, default=None):
            return self.knobs.get(name, default)
        def read_json(self, name, default=None):
            path = self.workdir / name
            return json.loads(path.read_text()) if path.exists() else default
        def read_text(self, name):
            return (self.workdir / name).read_text()
        def write_json(self, name, payload):
            (self.workdir / name).write_text(json.dumps(payload, default=str))
        def write_text(self, name, text):
            (self.workdir / name).write_text(text)
        def set_fact(self, name, value):
            self.facts[name] = value
        def exists(self, name):
            return (self.workdir / name).exists()

    ctx = Ctx()
    ctx.write_json("brief.json", brief_payload or {
        "name": NAME, "brief": "analyse plates", "requires_input": []})
    return ctx


def _strategy_json(inputs):
    return json.dumps({
        "considered": [{"approach": "a", "why_not": ""}],
        "chosen": "a", "why": "simpler", "will_not": [], "closest_existing": "none",
        "knobs": [{"name": "input_dir", "kind": "str", "default": "",
                   "description": "folder", "required": True}],
        "nodes": [{"id": "analyse", "kind": "op", "call": "ops:analyse",
                   "knobs": ["input_dir"], "writes": ["report.json"],
                   "produces": ["rows"]}],
        "edges": [], "start": "analyse", "score": "rows",
        "inputs": inputs,
        "findings": {},
    })


def test_reloop_second_plan_replaces_first(tmp_path):
    """On a reloop, a second strategy's inputs replace the first plan's."""
    from refactory_dsh.create_ops import _adopt_strategy_inputs

    ctx = _make_ctx(tmp_path)
    _adopt_strategy_inputs(ctx, _strategy_json(["input_dir"]))
    brief = ctx.read_json("brief.json", {})
    assert brief["requires_input"] == ["input_dir"]
    assert brief["requires_input_source"] == "plan"

    _adopt_strategy_inputs(ctx, _strategy_json(["query_text"]))
    brief = ctx.read_json("brief.json", {})
    assert brief["requires_input"] == ["query_text"], (
        "the second plan's inputs must replace the first plan's")
    assert brief["requires_input_source"] == "plan"


def test_input_flag_wins_over_plan_on_reloop(tmp_path):
    """--input (requires_input_source == '--input') blocks adoption on reloop."""
    from refactory_dsh.create_ops import _adopt_strategy_inputs

    ctx = _make_ctx(tmp_path, {
        "name": NAME, "brief": "analyse plates",
        "requires_input": ["user_file"],
        "requires_input_source": "--input",
    })
    _adopt_strategy_inputs(ctx, _strategy_json(["input_dir"]))
    brief = ctx.read_json("brief.json", {})
    assert brief["requires_input"] == ["user_file"], (
        "--input must win: plan cannot overwrite CLI-declared inputs")
    assert brief["requires_input_source"] == "--input"


# ── adopt_strategy_inputs: type safety on inputs ────────────────────────


def test_string_inputs_ignored_with_warning(tmp_path):
    """A string 'inputs' in the strategy is ignored (not split into chars)."""
    from refactory_dsh.create_ops import _adopt_strategy_inputs
    import warnings as _warnings

    ctx = _make_ctx(tmp_path)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _adopt_strategy_inputs(ctx, _strategy_json("input_dir"))
    brief = ctx.read_json("brief.json", {})
    assert not brief.get("requires_input"), (
        "a string 'inputs' must not be adopted (would split into characters)")
    assert any("must be a list" in str(w.message) for w in caught), (
        "a warning must be emitted for a non-list 'inputs'")


def test_dict_inputs_ignored_with_warning(tmp_path):
    """A dict 'inputs' in the strategy is ignored."""
    from refactory_dsh.create_ops import _adopt_strategy_inputs
    import warnings as _warnings

    ctx = _make_ctx(tmp_path)
    raw = json.dumps({
        "considered": [{"approach": "a", "why_not": ""}],
        "chosen": "a", "why": "simpler", "will_not": [], "closest_existing": "none",
        "knobs": [], "nodes": [], "edges": [], "start": "x", "score": "y",
        "inputs": {"key": "val"},
        "findings": {},
    })
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _adopt_strategy_inputs(ctx, raw)
    brief = ctx.read_json("brief.json", {})
    assert not brief.get("requires_input"), (
        "a dict 'inputs' must not be adopted")
    assert any("must be a list" in str(w.message) for w in caught)


def test_list_with_non_strings_filtered(tmp_path):
    """Non-string items in the inputs list are silently dropped."""
    from refactory_dsh.create_ops import _adopt_strategy_inputs

    ctx = _make_ctx(tmp_path)
    raw = json.dumps({
        "considered": [{"approach": "a", "why_not": ""}],
        "chosen": "a", "why": "simpler", "will_not": [], "closest_existing": "none",
        "knobs": [], "nodes": [], "edges": [], "start": "x", "score": "v",
        "inputs": ["good_name", 42, None, "", {"nested": True}],
        "findings": {},
    })
    _adopt_strategy_inputs(ctx, raw)
    brief = ctx.read_json("brief.json", {})
    assert brief.get("requires_input") == ["good_name"], (
        "only non-empty string items survive filtering")


# ── produced_results ─────────────────────────────────────────────────────


def test_produced_results_filters_empty_root_and_items_override(tmp_path):
    """The writes filter excludes files under the empty-input root and the items override,
    and reports only declared writes the run left behind."""
    from refactory_dsh.nn.executor import ITEMS_OVERRIDE

    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "report.json").write_text("{}")
    (workdir / "summary.csv").write_text("a,b\n1,2\n")
    empty_dir = workdir / empty_input.EMPTY_ROOT / "knob-x"
    empty_dir.mkdir(parents=True)
    (empty_dir / "placeholder.txt").write_text("")
    (workdir / ITEMS_OVERRIDE).write_text("{}")
    writes = ["report.json", "summary.csv", "missing.txt",
              f"{empty_input.EMPTY_ROOT}/knob-x/placeholder.txt", ITEMS_OVERRIDE]
    result = empty_input.produced_results(workdir, writes)
    assert result == ["report.json", "summary.csv"], (
        "only existing declared writes outside the empty root and items override are reported")
