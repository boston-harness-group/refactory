"""The empty-input check: a created mode that invents input when it has none is refused.

A created mode that found no input files generated a stand-in table, reported on it and
exited 0. After the canonical smoke passes, create mode now runs the mode once more with
its inputs pointed somewhere empty (`refactory_dsh.empty_input`), and a mode that then
finishes with results goes back to the builder.

The fixture modes are real packages run in the sandbox: one that fabricates a stand-in
when its input folder has no CSV files, one that stops with an error naming what is
missing, and the same pair reading a data node's source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import create as create_module
from refactory_dsh import empty_input, lowering, registry
from refactory_dsh.create import CreateConfig
from refactory_dsh.paths import project_for_nodes
from tests_support import FakeModel, create_script, good_files, render

NAME = "plate_pkg"

_INIT = '''\
"""Summarise a folder of plate CSVs."""
from refactory_dsh.nn import Package

from .mode import Plate

PACKAGE = Package(name="{name}", description="summarise plate CSVs", modes=(Plate,),
                  ops=("refactory_packages.{name}.ops",))
'''

_MODE = '''\
"""The plate summary mode."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Plate(nn.Mode):
    name = "plate"
    description = "summarise every CSV in the input folder"

    @classmethod
    def declare(cls):
        return {{"input_dir": nn.Knob("", "str", description="folder of plate CSVs")}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("summarise", "op", call=OPS + ":summarise", knobs=("input_dir",),
                          writes=("summary.json",), produces=("rows",)))
        return graph

    def score(self, run):
        return float(run.facts["rows"])
'''

_READ = '''\
from pathlib import Path


def _tables(ctx):
    folder = Path(str(ctx.knob("input_dir", "") or "."))
    return sorted(folder.glob("*.csv")) if folder.is_dir() else []


def _rows(paths):
    return sum(max(0, len(p.read_text().splitlines()) - 1) for p in paths)
'''

FABRICATES = _READ + '''

def summarise(ctx):
    tables = _tables(ctx)
    if not tables:
        stand_in = ctx.workdir / "work" / "synthetic_plate.csv"
        stand_in.parent.mkdir(parents=True, exist_ok=True)
        stand_in.write_text("well,value\\nA1,0.5\\nA2,0.5\\n")
        tables = [stand_in]
    rows = _rows(tables)
    ctx.write_json("summary.json", {"tables": [t.name for t in tables], "rows": rows})
    return {"rows": rows}
'''

REFUSES = _READ + '''

def summarise(ctx):
    tables = _tables(ctx)
    if not tables:
        raise ValueError(f"no input: input_dir {ctx.knob('input_dir', '')!r} holds no CSV files")
    rows = _rows(tables)
    ctx.write_json("summary.json", {"tables": [t.name for t in tables], "rows": rows})
    return {"rows": rows}
'''


def plate_files(name: str, ops: str) -> dict[str, str]:
    return {"__init__.py": _INIT.format(name=name), "mode.py": _MODE.format(name=name),
            "ops.py": ops}


#: The same two behaviours, with the input arriving through a data node's source.
_DATA_MODE = '''\
"""Count the items of a source."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Items(nn.Mode):
    name = "items"
    description = "count the items a source lists"

    @classmethod
    def declare(cls):
        return {{}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="seen", subgraph_exit="seen",
                          source_path="items", source_format="directory",
                          writes=("current_item.json",), produces=("item_scores",)))
        graph.add(nn.Node("seen", "op", call=OPS + ":seen", reads=("current_item.json",),
                          writes=("seen.json",), produces=("count",)))
        graph.add(nn.Node("finish", "op", call=OPS + ":finish", writes=("report.json",),
                          produces=("total",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run):
        return float(run.facts.get("total") or 0)
'''

_DATA_OPS = '''\
def seen(ctx):
    seen = ctx.read_json("seen.json", {{"ids": []}}) or {{"ids": []}}
    seen["ids"].append(str((ctx.read_json("current_item.json", {{}}) or {{}}).get("id")))
    ctx.write_json("seen.json", seen)
    return {{"count": len(seen["ids"])}}


def finish(ctx):
    ids = (ctx.read_json("seen.json", {{"ids": []}}) or {{"ids": []}})["ids"]
{empty}
    ctx.write_json("report.json", {{"total": len(ids)}})
    return {{"total": len(ids)}}
'''


def data_files(name: str, *, refuses: bool) -> dict[str, str]:
    empty = ('    if not ids:\n        raise ValueError("no input: the data source listed no items")'
             if refuses else '    if not ids:\n        ids = ["sampled-1", "sampled-2"]')
    init = _INIT.replace("from .mode import Plate", "from .mode import Items").replace(
        "modes=(Plate,)", "modes=(Items,)")
    return {"__init__.py": init.format(name=name), "mode.py": _DATA_MODE.format(name=name),
            "ops.py": _DATA_OPS.format(empty=empty)}


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("REFACTORY_PROJECT", str(project))
    monkeypatch.chdir(project)
    return project


@pytest.fixture
def plates(tmp_path) -> Path:
    """The person's real input: a folder holding one plate CSV."""
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


def _run(project: Path, tmp_path: Path, identifier: str, samples: dict[str, str]) -> dict:
    return empty_input.run(identifier, tmp_path / "empty-run", project=project,
                           samples=samples, declared=(), model=False, timeout_s=120)


def test_a_mode_that_fabricates_input_is_a_finding(tmp_path, project, plates):
    _install(project, NAME, plate_files(NAME, FABRICATES))
    receipt = _run(project, tmp_path, f"{NAME}/plate", {"input_dir": str(plates)})
    assert receipt["exit_ok"] is True
    assert receipt["status"] == "finding", receipt
    assert receipt["produced"] == ["summary.json"]
    assert "must not fabricate" in receipt["reason"]
    # The real input was replaced by an empty folder, never read.
    emptied = Path(receipt["knobs"]["input_dir"])
    assert emptied.is_dir() and not any(emptied.iterdir())


def test_a_mode_that_stops_on_empty_input_passes(tmp_path, project, plates):
    _install(project, NAME, plate_files(NAME, REFUSES))
    receipt = _run(project, tmp_path, f"{NAME}/plate", {"input_dir": str(plates)})
    assert receipt["status"] == "pass", receipt
    assert receipt["exit_ok"] is False
    assert "holds no CSV files" in receipt["error"]
    assert receipt["produced"] == []


@pytest.mark.parametrize("refuses, status", [(False, "finding"), (True, "pass")])
def test_a_data_node_source_is_emptied_too(tmp_path, project, refuses, status):
    _install(project, "items_pkg", data_files("items_pkg", refuses=refuses))
    receipt = _run(project, tmp_path, "items_pkg/items", {})
    assert receipt["sources"]["data"]["source_format"] == "directory"
    assert receipt["status"] == status, receipt


def test_a_declared_input_knob_is_emptied_without_a_sample(tmp_path, project):
    _install(project, NAME, plate_files(NAME, FABRICATES))
    receipt = empty_input.run(f"{NAME}/plate", tmp_path / "empty-run", project=project,
                              samples={}, declared=("input_dir",), model=False, timeout_s=120)
    assert receipt["status"] == "finding", receipt


def test_a_mode_with_no_input_is_not_applicable(tmp_path, project):
    _install(project, "counter_pkg", good_files("counter_pkg"))
    receipt = _run(project, tmp_path, "counter_pkg/counter", {})
    assert receipt["status"] == "not_applicable"


@pytest.mark.parametrize("kwargs, status", [
    ({"ok": True, "timed_out": False, "error": "", "produced": ["out.json"]}, "finding"),
    ({"ok": True, "timed_out": False, "error": "", "produced": []}, "pass"),
    ({"ok": False, "timed_out": False, "error": "ValueError: no input", "produced": []}, "pass"),
    ({"ok": False, "timed_out": True, "error": "", "produced": ["out.json"]}, "inconclusive"),
    ({"ok": False, "timed_out": False, "error": "model completion failed", "produced": []},
     "inconclusive"),
])
def test_the_verdict(kwargs, status):
    assert empty_input.verdict(**kwargs)[0] == status


# ── wired into create mode's smoke gate ───────────────────────────────────


def test_create_sends_a_fabricating_mode_back_to_the_builder(tmp_path, project, plates):
    """The full create graph on the walker: the first package fabricates, the smoke gate's
    empty-input run finds it and reloops to emit with the reason, the second package
    stops on empty input and is accepted."""
    model = FakeModel(create_script(NAME, [render(plate_files(NAME, FABRICATES)),
                                           render(plate_files(NAME, REFUSES))]))
    result = create_module.create(
        CreateConfig(verdicts={"gate_strategy": "proceed"}, single_shot=True, name=NAME,
                     brief="summarise plate CSVs", attempts=3, workdir=tmp_path / "run",
                     samples=[f"input_dir={plates}"]),
        client=model,
    )
    emit_prompts = [prompt for prompt in model.prompts if "Write the package" in prompt]
    assert len(emit_prompts) == 2, "the fabricating package must be sent back once"
    assert "empty-input check" in emit_prompts[-1]
    assert "must not fabricate" in emit_prompts[-1]
    created = json.loads((tmp_path / "run" / "create.json").read_text())
    assert created["empty_input_check"]["status"] == "pass"
    assert result.smoke_score == 3.0
    assert "def summarise" in (registry.workspace_root(project_for_nodes()) / NAME
                               / "ops.py").read_text()


@pytest.mark.parametrize("ops, verdict, status", [
    (FABRICATES, "reloop", "finding"), (REFUSES, "proceed", "pass"),
], ids=["fabricates", "refuses"])
def test_the_smoke_gate_runs_the_check_as_the_spine_invokes_it(tmp_path, project, plates,
                                                                ops, verdict, status):
    """The spine runs the smoke gate as `python -m refactory_dsh.run ...:gate_smoke` in the
    run directory, with the lowered declarations and the knobs in the environment."""
    import os
    import subprocess
    import sys

    from refactory_dsh.create import CreateMode
    from refactory_dsh.lowering import DECLARATIONS, KNOBS_ENV
    from refactory_dsh.run import FACTS, NODE_ENV

    _install(project, NAME, plate_files(NAME, ops))
    run = tmp_path / "run"
    run.mkdir()
    node = CreateMode().build().nodes["smoke"]
    # What `prepare` leaves for the later steps.
    (run / "brief.json").write_text(json.dumps({"name": NAME, "brief": "summarise plates",
                                                "requires_input": []}))
    (run / FACTS).write_text(json.dumps({
        "name": NAME, "package_path": str(registry.workspace_root(project) / NAME)}))
    (run / DECLARATIONS).parent.mkdir(parents=True, exist_ok=True)
    (run / DECLARATIONS).write_text(json.dumps({"smoke": {
        "reads": [], "writes": list(node.writes), "produces": list(node.produces)}}))
    env = {**os.environ, "REFACTORY_PROJECT": str(project), "REFACTORY_WORKDIR": str(run),
           KNOBS_ENV: json.dumps({"sample_knobs": json.dumps([f"input_dir={plates}"]),
                                  "smoke_timeout_s": 120, "attempts": 3}),
           NODE_ENV: json.dumps({"id": "smoke", "type": "GateNode", "reads": [],
                                 "writes": list(node.writes)})}
    done = subprocess.run(
        [sys.executable, "-m", "refactory_dsh.run", "refactory_dsh.create_ops:gate_smoke"],
        cwd=str(run), env=env, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == verdict, done.stdout
    assert json.loads((run / empty_input.RECEIPT).read_text())["status"] == status
    if verdict == "reloop":
        assert "empty-input check" in done.stdout


@pytest.mark.parametrize("mode", ["create", "edit", "create_fast", "edit_fast"])
def test_every_create_graph_lowers_the_check_into_its_smoke_gate(mode):
    """Both runtimes run the smoke gate's own op; the lowered gate (the spine's) declares
    the receipt and has room for the third sandboxed run."""
    from refactory_dsh.create import CreateMode
    from refactory_dsh.edit import EditMode
    from refactory_dsh.fast import CreateFastMode, EditFastMode

    cls = {"create": CreateMode, "edit": EditMode, "create_fast": CreateFastMode,
           "edit_fast": EditFastMode}[mode]
    state = cls().state_dict()
    workflow = lowering.to_workflow(state, name=f"default_pack-{mode}")
    smoke = workflow.nodes["smoke"]
    assert empty_input.RECEIPT in smoke.writes
    assert smoke.evaluator_command.endswith("refactory_dsh.create_ops:gate_smoke")
    budget = int(state["knobs"]["smoke_timeout_s"]["value"])
    assert smoke.command_timeout_s >= 3 * budget
