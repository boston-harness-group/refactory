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
from refactory_dsh import create_ops, empty_input, lowering, registry
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


#: Says there was no input in a status report instead of raising: an honest exit 0.
REPORTS_NONE = _READ + '''

def summarise(ctx):
    tables = _tables(ctx)
    if not tables:
        ctx.write_json("summary.json", {"status": "no input",
                                        "reason": "input_dir holds no CSV files",
                                        "tables": [], "rows": 0})
        return {"rows": 0}
    rows = _rows(tables)
    ctx.write_json("summary.json", {"tables": [t.name for t in tables], "rows": rows})
    return {"rows": rows}
'''


def plate_files(name: str, ops: str, *, knobs: str = "", writes: str = "summary.json"
                ) -> dict[str, str]:
    """The plate package; ``knobs`` adds declarations, ``writes`` renames its result."""
    mode = _MODE.format(name=name).replace(
        'description="folder of plate CSVs")', 'description="folder of plate CSVs")' + knobs)
    if knobs:
        mode = mode.replace('knobs=("input_dir",)', 'knobs=("input_dir", "config", "reference")')
    return {"__init__.py": _INIT.format(name=name),
            "mode.py": mode.replace("summary.json", writes),
            "ops.py": ops.replace("summary.json", writes)}


#: A plate mode with a configuration file beside its input, which it must read.
_CONFIG_KNOBS = (',\n                "config": nn.Knob("", "str", description="settings file"),'
                 '\n                "reference": nn.Knob("", "str", description="unused")')
CONFIG_FABRICATES = FABRICATES.replace('''def summarise(ctx):
    tables = _tables(ctx)''', '''def summarise(ctx):
    import json
    json.loads(Path(str(ctx.knob("config", ""))).read_text())["threshold"]
    tables = _tables(ctx)''')


#: Counts the words of a text knob into an output folder and a declared report.
_TEXT_MODE = '''\
"""Count the words of a text."""
from refactory_dsh import nn

OPS = "refactory_packages.{name}.ops"


class Words(nn.Mode):
    name = "words"
    description = "count the words of a text"

    @classmethod
    def declare(cls):
        return {{"text": nn.Knob("hello there", "str", description="the text"),
                 "out_dir": nn.Knob("", "str", description="where the counts go")}}

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("count", "op", call=OPS + ":count", knobs=("text", "out_dir"),
                          writes=("report.json",), produces=("words",)))
        return graph

    def score(self, run):
        return float(run.facts["words"])
'''

_TEXT_OPS = '''\
import json
from pathlib import Path


def count(ctx):
    text = str(ctx.knob("text", ""))
    if not text.strip():
        raise ValueError("no input: the text is empty")
    words = len(text.split())
    out = Path(str(ctx.knob("out_dir", "") or ctx.workdir))
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps({"words": words}))
    ctx.write_json("report.json", {"words": words})
    return {"words": words}
'''


def text_files(name: str) -> dict[str, str]:
    init = _INIT.replace("from .mode import Plate", "from .mode import Words").replace(
        "modes=(Plate,)", "modes=(Words,)")
    return {"__init__.py": init.format(name=name), "mode.py": _TEXT_MODE.format(name=name),
            "ops.py": _TEXT_OPS}


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


class Ctx:
    """The parts of a run context the smoke gate's steps use, in a run directory."""

    def __init__(self, workdir: Path, facts: dict):
        self.workdir, self.facts, self.knobs = workdir, dict(facts), {}
        self.model = self.budget = None

    def knob(self, name: str, default=None):
        return self.knobs.get(name, default)

    def read_json(self, name: str, default=None):
        path = self.workdir / name
        return json.loads(path.read_text()) if path.exists() else default

    def write_json(self, name: str, payload) -> None:
        (self.workdir / name).write_text(json.dumps(payload, default=str))


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


def _run(project: Path, tmp_path: Path, identifier: str, samples: dict[str, str],
         declared: tuple[str, ...] = ("input_dir",)) -> dict:
    return empty_input.run(identifier, tmp_path / "empty-run", project=project,
                           samples=samples, declared=declared, model=False, timeout_s=120)


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
    """No sample: the declared knob's default (empty here) becomes the empty string."""
    _install(project, NAME, plate_files(NAME, FABRICATES))
    receipt = empty_input.run(f"{NAME}/plate", tmp_path / "empty-run", project=project,
                              samples={}, declared=("input_dir",), model=False, timeout_s=120)
    assert receipt["status"] == "finding", receipt


def test_a_mode_with_no_input_is_not_applicable(tmp_path, project):
    _install(project, "counter_pkg", good_files("counter_pkg"))
    receipt = _run(project, tmp_path, "counter_pkg/counter", {})
    assert receipt["status"] == "not_applicable"
    assert "not applicable" in empty_input.warning(receipt)


# ── only declared input is emptied (H1, M2) ───────────────────────────────


def test_an_output_folder_sample_is_never_treated_as_input(tmp_path, project):
    """The words mode writes into an `out_dir` sample. That folder is not input: with no
    declared input the check is not applicable, and with its text declared the empty text
    makes the mode stop, which passes. Neither is a finding."""
    _install(project, "words_pkg", text_files("words_pkg"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    samples = {"text": "three short words", "out_dir": str(out_dir)}
    undeclared = _run(project, tmp_path / "a", "words_pkg/words", samples, declared=())
    assert undeclared["status"] == "not_applicable", undeclared
    declared = _run(project, tmp_path / "b", "words_pkg/words", samples, declared=("text",))
    assert declared["status"] == "pass", declared
    assert declared["knobs"] == {"text": ""}, "only the declared input is emptied"
    assert "the text is empty" in declared["error"]


def test_a_fabricating_mode_with_a_configuration_sample_is_still_caught(tmp_path, project,
                                                                         plates):
    """The configuration file is not input. Emptying it would make the mode crash on its
    settings and pass; kept, the mode reaches its empty input and fabricates."""
    _install(project, NAME, plate_files(NAME, CONFIG_FABRICATES, knobs=_CONFIG_KNOBS))
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"threshold": 0.2}))
    receipt = _run(project, tmp_path, f"{NAME}/plate",
                   {"input_dir": str(plates), "config": str(config)})
    assert receipt["status"] == "finding", receipt
    assert "config" not in receipt["knobs"]


def test_an_undeclared_knob_defaulting_to_real_data_is_not_run(tmp_path, project, plates):
    """A knob the run does not declare as input, whose default names existing data, would
    still be read with the declared input emptied. The check does not run the mode on it:
    it is inconclusive and names the knob."""
    knobs = _CONFIG_KNOBS.replace('"reference": nn.Knob("", ',
                                  f'"reference": nn.Knob({str(plates)!r}, ')
    _install(project, NAME, plate_files(NAME, FABRICATES, knobs=knobs))
    receipt = _run(project, tmp_path, f"{NAME}/plate", {"input_dir": str(plates)})
    assert receipt["status"] == "inconclusive", receipt
    assert receipt["undeclared"] == [f"reference (default {plates})"]
    assert "exit_ok" not in receipt, "the mode must not be run on its real data"
    assert not (tmp_path / "empty-run" / "summary.json").exists()
    assert "inconclusive" in empty_input.warning(receipt)


def test_a_declared_input_that_the_graph_writes_is_left_alone(tmp_path):
    described = {"knobs": ["input_dir", "report"], "values": {"report": "report.json"},
                 "data_nodes": [], "writes": ["report.json"]}
    planned = empty_input.plan(described, {}, ("report",), empty_root=tmp_path / "e",
                               project=tmp_path)
    assert planned["knobs"] == {}
    assert "names a file the graph writes" in planned["kept"][0]


# ── an honest report of no input passes (M1) ──────────────────────────────


def test_a_mode_that_reports_no_input_passes(tmp_path, project, plates):
    _install(project, NAME, plate_files(NAME, REPORTS_NONE))
    receipt = _run(project, tmp_path, f"{NAME}/plate", {"input_dir": str(plates)})
    assert receipt["exit_ok"] is True
    assert receipt["status"] == "pass", receipt
    assert receipt["empty_outputs"] == ["summary.json"] and receipt["results"] == []


@pytest.mark.parametrize("name, content, empty", [
    ("a.txt", "", True),
    ("a.txt", "  \n", True),
    ("a.json", "[]", True),
    ("a.json", '{"count": 0, "items": []}', True),
    ("a.json", '{"status": "no input", "reason": "the folder is empty", "rows": 0}', True),
    ("a.jsonl", "\n{}\n", True),
    ("a.csv", "well,value\n", True),
    ("a.json", '{"rows": 2}', False),
    ("a.json", '{"tables": ["synthetic.csv"]}', False),
    ("a.json", "not json", False),
    ("a.jsonl", '{"id": "x"}\n', False),
    ("a.csv", "well,value\nA1,0.5\n", False),
    ("a.txt", "mean 0.5", False),
])
def test_what_counts_as_an_empty_output(tmp_path, name, content, empty):
    path = tmp_path / name
    path.write_text(content)
    assert empty_input.is_empty_output(path) is empty


def test_a_folder_output_is_empty_only_when_every_file_is(tmp_path):
    folder = tmp_path / "out"
    (folder / "nested").mkdir(parents=True)
    (folder / "nested" / "rows.csv").write_text("well,value\n")
    assert empty_input.is_empty_output(folder)
    (folder / "result.json").write_text('{"rows": 1}')
    assert not empty_input.is_empty_output(folder)


@pytest.mark.parametrize("kwargs, status", [
    ({"ok": True, "timed_out": False, "error": "", "results": ["out.json"]}, "finding"),
    ({"ok": True, "timed_out": False, "error": "", "results": [], "empty": ["out.json"]},
     "pass"),
    ({"ok": True, "timed_out": False, "error": "", "results": []}, "pass"),
    ({"ok": False, "timed_out": False, "error": "ValueError: no input", "results": []}, "pass"),
    ({"ok": False, "timed_out": True, "error": "", "results": ["out.json"]}, "inconclusive"),
    ({"ok": False, "timed_out": False, "error": "model completion failed", "results": []},
     "inconclusive"),
])
def test_the_verdict(kwargs, status):
    assert empty_input.verdict(**kwargs)[0] == status


# ── an unjudged check is inconclusive and said; the budget is shared (M3, L4) ──


def test_a_describe_that_fails_is_inconclusive_not_a_failure(tmp_path, project, monkeypatch):
    _install(project, NAME, plate_files(NAME, FABRICATES))

    def stuck(*_args, **_kwargs):
        raise ValueError("the mode could not be described: timed out after 120s")

    monkeypatch.setattr(empty_input, "describe", stuck)
    receipt = _run(project, tmp_path, f"{NAME}/plate", {})
    assert receipt["status"] == "inconclusive"
    assert "could not be described" in receipt["reason"]


def test_describe_and_the_run_share_one_smoke_budget(tmp_path, monkeypatch):
    """Describing gets at most what the smoke budget allows, and the run what is left, so
    the three sandboxed runs the lowered smoke gate's envelope covers stay three."""
    from refactory_dsh import sandbox

    seen: dict[str, int] = {}
    described = {"knobs": ["input_dir"], "values": {"input_dir": ""}, "data_nodes": [],
                 "writes": []}

    def describe(_identifier, _project, *, timeout_s):
        seen["describe"] = timeout_s
        return described

    def smoke(*_args, timeout_s, **_kwargs):
        seen["run"] = timeout_s
        return sandbox.SandboxResult(ok=False, timed_out=True)

    clock = iter([1000.0, 1050.0])
    monkeypatch.setattr(empty_input, "describe", describe)
    monkeypatch.setattr(sandbox, "smoke", smoke)
    monkeypatch.setattr(empty_input.time, "monotonic", lambda: next(clock))
    empty_input.run("p/m", tmp_path / "run", project=tmp_path, samples={},
                    declared=("input_dir",), model=False, timeout_s=100)
    assert seen == {"describe": 100, "run": 50}

    seen.clear()
    clock = iter([1000.0, 1100.0])
    receipt = empty_input.run("p/m", tmp_path / "run2", project=tmp_path, samples={},
                              declared=("input_dir",), model=False, timeout_s=100)
    assert seen == {"describe": 100}, "no budget was left for the run"
    assert receipt["status"] == "inconclusive"


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
                     requires_input="input_dir", samples=[f"input_dir={plates}"]),
        client=model,
    )
    emit_prompts = [prompt for prompt in model.prompts if "Write the package" in prompt]
    assert len(emit_prompts) == 2, "the fabricating package must be sent back once"
    assert "empty-input check" in emit_prompts[-1]
    assert "must not fabricate" in emit_prompts[-1]
    created = json.loads((tmp_path / "run" / "create.json").read_text())
    assert created["empty_input_check"]["status"] == "pass"
    assert not any("empty-input" in w for w in created.get("warnings", []))
    assert result.smoke_score == 3.0
    assert "def summarise" in (registry.workspace_root(project_for_nodes()) / NAME
                               / "ops.py").read_text()


def test_a_create_with_no_declared_input_warns_on_the_walker(tmp_path, project, plates):
    """No `--input`: the check is not applicable. The walker journals it as a run warning
    and the run's report lists it, so a person sees the check did not judge the mode."""
    from refactory_dsh import live

    model = FakeModel(create_script(NAME, [render(plate_files(NAME, REFUSES))]))
    create_module.create(
        CreateConfig(verdicts={"gate_strategy": "proceed"}, single_shot=True, name=NAME,
                     brief="summarise plate CSVs", attempts=3, workdir=tmp_path / "run",
                     samples=[f"input_dir={plates}"]),
        client=model,
    )
    created = json.loads((tmp_path / "run" / "create.json").read_text())
    assert created["empty_input_check"]["status"] == "not_applicable"
    assert any("empty-input check was not applicable" in w for w in created["warnings"])
    records, _ = live.events(tmp_path / "run")
    assert any(r.get("kind") == "run_warning" and "empty-input check" in r.get("summary", "")
               for r in records)


@pytest.mark.parametrize("ops, writes, declared, verdict, status", [
    (FABRICATES, "summary.json", ["input_dir"], "reloop", "finding"),
    # The finding names the file the mode wrote. Words in it that read as an outage
    # ("credentials") must not turn a repairable finding into an unrepairable stop.
    (FABRICATES, "credentials.json", ["input_dir"], "reloop", "finding"),
    (REFUSES, "summary.json", ["input_dir"], "proceed", "pass"),
    (FABRICATES, "summary.json", [], "proceed", "not_applicable"),
], ids=["fabricates", "finding-names-credentials", "refuses", "nothing-declared"])
def test_the_smoke_gate_runs_the_check_as_the_spine_invokes_it(tmp_path, project, plates, ops,
                                                                writes, declared, verdict,
                                                                status):
    """The spine runs the smoke gate as `python -m refactory_dsh.run ...:gate_smoke` in the
    run directory, with the lowered declarations and the knobs in the environment. An
    unjudged check proceeds with a run warning."""
    import os
    import subprocess
    import sys

    from refactory_dsh.create import CreateMode
    from refactory_dsh.lowering import DECLARATIONS, KNOBS_ENV
    from refactory_dsh.run import FACTS, NODE_ENV

    _install(project, NAME, plate_files(NAME, ops, writes=writes))
    run = tmp_path / "run"
    run.mkdir()
    node = CreateMode().build().nodes["smoke"]
    # What `prepare` leaves for the later steps.
    (run / "brief.json").write_text(json.dumps({"name": NAME, "brief": "summarise plates",
                                                "requires_input": declared}))
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
        evidence = json.loads((run / "authoring-evidence.json").read_text())
        assert evidence["validation"]["status"] == "code_failure"
        assert evidence["validation"]["empty_input"]["status"] == "finding"
    warned = "run warning: the empty-input check was not applicable" in done.stderr
    assert warned is (status == "not_applicable"), done.stderr


def test_a_finding_unmarks_a_package_that_had_passed(tmp_path, project, plates):
    """`check_empty_input` on its own: a package recorded as passed is pending again once
    the check finds it fabricating, whoever called it."""

    _install(project, NAME, plate_files(NAME, FABRICATES))
    package = registry.workspace_root(project) / NAME
    registry.record_smoke(package, registry.SMOKE_PASSED)
    ctx = Ctx(tmp_path / "run", {"name": NAME, "package_path": str(package)})
    ctx.workdir.mkdir()
    ctx.write_json("smoke.json", {"mode": f"{NAME}/plate"})
    ctx.write_json("brief.json", {"requires_input": ["input_dir"]})
    ctx.knobs.update({"sample_knobs": json.dumps([f"input_dir={plates}"]),
                      "smoke_timeout_s": 120})
    with pytest.raises(empty_input.Finding) as raised:
        create_ops.check_empty_input(ctx)
    assert raised.value.receipt["status"] == "finding"
    assert registry.recorded_smoke(package) == registry.SMOKE_PENDING


def test_the_report_carries_an_unjudged_check_as_a_warning(tmp_path, project, monkeypatch):
    from refactory_dsh import authoring_evidence

    _install(project, NAME, plate_files(NAME, FABRICATES))
    package = registry.workspace_root(project) / NAME
    (package / "README.md").write_text("# plates\n")
    ctx = Ctx(tmp_path / "run", {"name": NAME, "package_path": str(package)})
    ctx.workdir.mkdir()
    monkeypatch.setattr(authoring_evidence, "require_completion", lambda _ctx: None)
    ctx.write_json(empty_input.RECEIPT, {"mode": f"{NAME}/plate", "status": "inconclusive",
                                         "reason": "the empty-input run was stopped"})
    created = json.loads(create_ops.write_result(ctx)["created"])
    assert created["warnings"] == [
        f"the empty-input check was inconclusive for {NAME}/plate: the empty-input run "
        "was stopped"]
    ctx.write_json(empty_input.RECEIPT, {"mode": f"{NAME}/plate", "status": "pass"})
    assert "warnings" not in json.loads(create_ops.write_result(ctx)["created"])


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
