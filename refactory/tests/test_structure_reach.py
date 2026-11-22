"""New steps reach the item's files and wire themselves in, without any way to cheat.

- **Item files** (`Task.item_files`): a Task lays files into each item's pass under fixed
  names; the data node declares them (V12); an op opens one only through
  `RunContext.item_file`, which refuses a file the op does not declare; a sealed step
  that reads one gets a copy and nothing else; a golden file can never be one. Run on the
  walker (sequential and concurrent passes) and on the TS spine (`items.command_begin`,
  the spine's op runner and its agent dispatch through `worker_node`).
- **`to` an agent whose prompt is no knob** re-slots its prompt in place, in every graph
  shape (fork branch, nested fork, loop, data pass), each held to both runtimes.
- **The request** says what an existing step's tool rules mean for a new step, shows the
  item files and who reads them, and its operator table stays short in a loop and agrees
  with an attempt.
- **`new_step_tools`** off refuses every new step with tools and offers none.

Each test names the runtimes it covers in its docstring.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

import item_files_ops as ifo
import structure_fuzz as sf
from refactory_dsh import lowering, workers
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn import pipeline_view as pv
from refactory_dsh.nn.executor import ExecutionError, RunContext
from refactory_dsh.nn.graph import Graph, op_io
from refactory_dsh.nn.graph import Node as nn_node
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim import base as base_mod
from refactory_dsh.task import (Task, TaskError, check_laid_item_files, checked_item_files,
                                task_item_files)
from refactory_packages.spreadsheetbench import mode as sb_mode
from refactory_packages.spreadsheetbench import tasks as sb_tasks
from test_livemath_blocks import _spine, needs_spine

RUNTIMES = ["walker", pytest.param("spine", marks=needs_spine)]


def _records(mode_cls: type, runtime: str, tmp_path: Path, *, model=None, transform=None,
             parallelism: int = 1) -> tuple[list[dict], bool]:
    """``mode_cls``'s item records on one runtime, and whether the run ended well."""
    mode = mode_cls()
    if parallelism != 1:
        mode.load_state_dict(nn_edits.configure(mode.state_dict(), "data_parallelism",
                                                parallelism))
    if runtime == "walker":
        result = mode.run(tmp_path / "walker", model=model)
        return list(result.facts.get("item_scores", [])), bool(result.exit_ok)
    state = mode.state_dict()
    work = tmp_path / "spine"
    rc = _spine(mode.build(), work, tmp_path, mode_state=state,
                agent_responses={} if transform is not None else None,
                **({"transform": transform(state)} if transform is not None else {}))
    return ifo.records(work / "facts.json"), rc == 0


# ── A. item files ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("parallelism", [1, 2])
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_op_reads_its_declared_item_file(tmp_path, runtime, parallelism):
    """[walker, sequential and concurrent; spine] The Task lays each item's input into its
    pass; the op that declares it opens it through `RunContext.item_file`."""
    if runtime == "spine" and parallelism != 1:
        pytest.skip("the spine drain's concurrency is the walker's concern here")
    records, ok = _records(ifo.ItemFilesMode, runtime, tmp_path, parallelism=parallelism)
    assert ok
    assert sorted((r["id"], r["score"]) for r in records) == [("a", 1.0), ("b", 1.0)], records


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_op_that_does_not_declare_an_item_file_is_refused_it(tmp_path, runtime):
    """[walker, spine] Reading an item file around the declaration fails the item loudly,
    naming the file and the node's declared reads."""
    records, ok = _records(ifo.UndeclaredRead, runtime, tmp_path)
    assert not ok
    assert all(r["score"] == 0.0 for r in records), records
    assert "reads item file 'item/input.bin' (input), which it does not declare" in \
        str(records[0].get("error_text")), records


@pytest.mark.parametrize("parallelism", [1, 2])
@pytest.mark.parametrize("mode_cls,says", [(ifo.GoldLink, "is reached through a link"),
                                           (ifo.GoldHardLink, "is a hard link")],
                         ids=["symlink", "hard link"])
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_item_file_laid_as_a_link_to_the_answer_fails_the_item(tmp_path, runtime, mode_cls,
                                                                 says, parallelism):
    """[walker, sequential and concurrent; spine] A Task whose setup links an item file
    (symbolically or hard) to the answer outside the pass is refused before any step runs;
    the answer reaches no record."""
    if runtime == "spine" and parallelism != 1:
        pytest.skip("the spine's pass begins in `items.command_begin` whatever the concurrency")
    records, ok = _records(mode_cls, runtime, tmp_path, parallelism=parallelism)
    assert not ok
    assert all(r["score"] == 0.0 for r in records), records
    assert says in str(records[0].get("error_text")), records
    # No step ran on it: nothing was answered, and no error repeats the answer.
    assert not list(tmp_path.glob(f"{runtime}/passes/*/*/{ifo.ANSWER}"))
    assert ifo.SECRET not in json.dumps([r.get("error_text") for r in records])


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_reading_an_undeclared_item_file_by_name_is_refused_too(tmp_path, runtime):
    """[walker, spine] ``ctx.read_text`` (and ``read_json``, ``exists``) of an item file
    the op does not declare fails the item as `item_file` does; declared, it is read."""
    records, ok = _records(ifo.UndeclaredReadText, runtime, tmp_path / "undeclared")
    assert not ok
    assert "reads item file 'item/input.bin' (input), which it does not declare" in \
        str(records[0].get("error_text")), records
    records, ok = _records(ifo.DeclaredReadText, runtime, tmp_path / "declared")
    assert ok and sorted(r["score"] for r in records) == [1.0, 1.0], records


def test_every_read_helper_checks_item_files(tmp_path):
    """[the context both runtimes build] read_text, read_json and exists all refuse."""
    from refactory_dsh.items import PASS_MARKER

    (tmp_path / PASS_MARKER).parent.mkdir(parents=True)
    (tmp_path / PASS_MARKER).write_text(json.dumps({"task": f"{ifo.MODULE}:FileTask"}))
    (tmp_path / ifo.INPUT).parent.mkdir()
    (tmp_path / ifo.INPUT).write_text("{}")
    ctx = RunContext(tmp_path)
    ctx.node_id = "peek"
    for read in (ctx.read_text, ctx.read_json, ctx.exists):
        with pytest.raises(ExecutionError, match="which it does not declare"):
            read(ifo.INPUT)
        with pytest.raises(ExecutionError, match="which it does not declare"):
            read(f"./{ifo.INPUT}")
    ctx._declared_reads = (ifo.INPUT,)  # noqa: SLF001
    assert ctx.read_json(ifo.INPUT) == {} and ctx.exists(ifo.INPUT)


def test_item_file_accessor_refuses_outside_a_pass_and_unknown_roles(tmp_path):
    """[both runtimes build this context: `run.build_op_context`] No pass, no item file;
    a role the Task does not declare is named as unknown."""
    ctx = RunContext(tmp_path)
    ctx.node_id = "read"
    with pytest.raises(ExecutionError, match="outside an item's pass"):
        ctx.item_file("input")
    from refactory_dsh.items import PASS_MARKER

    (tmp_path / PASS_MARKER).parent.mkdir(parents=True)
    (tmp_path / PASS_MARKER).write_text(json.dumps({"task": f"{ifo.MODULE}:FileTask"}))
    ctx._declared_reads = (ifo.INPUT,)  # noqa: SLF001
    with pytest.raises(ExecutionError, match="does not lay in; it lays in: input"):
        ctx.item_file("gold")
    assert ctx.item_file("input") is None  # this item has none laid in
    (tmp_path / ifo.INPUT).parent.mkdir()
    (tmp_path / ifo.INPUT).write_text("x")
    assert ctx.item_file("input") == tmp_path / ifo.INPUT


@pytest.mark.parametrize("declared,says", [
    ({"input": "/abs/input.xlsx"}, "not a normalised path"),
    ({"input": "../input.xlsx"}, "not a normalised path"),
    ({"input": "item/./input.xlsx"}, "not a normalised path"),
    ({"input": ".refactory/input.xlsx"}, "runtime's own files"),
    ({"input": "current_item.json"}, "runtime's own files"),
    ({"input": "facts.json"}, "runtime's own files"),
    ({"input": "answer.txt"}, "the grader reads or writes"),
    ({"input": "item/answer.txt"}, "the grader reads or writes"),
    ({"input": "grade.json"}, "the grader reads or writes"),
    ({"input": "item", "other": "item/x"}, "folder of another item file"),
    ({"in put": "item/x"}, "is not a word"),
    ({"input": "Answer.txt"}, "the grader reads or writes"),
    ({"input": "item/\uff41\uff4e\uff53\uff57\uff45\uff52.txt"}, "the grader reads or writes"),
    ({"input": "item/GRADE.JSON"}, "the grader reads or writes"),
    ({"input": "item/ans\u200bwer.txt"}, "control or invisible character"),
    ({"input": "Current_Item.json"}, "runtime's own files"),
    ({"input": ".Refactory/x"}, "runtime's own files"),
])
def test_an_item_file_is_never_a_file_the_grader_or_runtime_owns(declared, says):
    """[declaration; both runtimes and the validator read it through here] A golden or graded
    file, a grader output, the runtime's records or a path out of the pass is refused."""
    class Bad(Task):
        name = "bad"
        graded_files = ("answer.txt",)
        grader_outputs = ("grade.json",)
        item_files = declared

    with pytest.raises(TaskError, match=says):
        checked_item_files(Bad)


def test_item_files_are_a_class_declaration_read_one_way(tmp_path):
    """[declaration] An instance cannot declare other item files than its class's: the
    validator, the view and both runtimes read the same ones."""
    task = ifo.FileTask()
    assert checked_item_files(task) == checked_item_files(ifo.FileTask) == {"input": ifo.INPUT}
    task.item_files = {"input": "answer.txt"}
    with pytest.raises(TaskError, match="class attributes"):
        checked_item_files(task)
    with pytest.raises(TaskError, match="class attributes"):
        check_laid_item_files(task, tmp_path)


def test_a_golden_file_cannot_be_reached_through_spreadsheetbench_item_files(tmp_path):
    """[declaration and pass contents; the runtimes copy only these files into a sealed step]
    SpreadsheetBench lays in only the input, under a fixed name; the golden file is in no
    item file and in no file of the pass, and a sealed step reading anything outside the
    pass is refused."""
    from sb_support import make_dataset

    from refactory_dsh import workspace

    files = task_item_files(sb_mode.TASK)
    assert files == {"input": "item/input.xlsx"} == dict(sb_tasks.SpreadsheetBenchTask.item_files)
    items = [{"id": "t1", "instruction": "Put 42 in B1.", "instruction_type": "Cell-Level Manipulation",
              "answer_position": "B1", "init": [{"title": "S", "cells": {"A1": 1}}],
              "gold": [{"title": "S", "cells": {"A1": 1, "B1": 42}}]}]
    root = tmp_path / "dataset"
    make_dataset(root, items, {"train": ["t1"], "val": [], "test": []})
    task = sb_tasks.SpreadsheetBenchTask.__new__(sb_tasks.SpreadsheetBenchTask)
    instance = sb_tasks.item_record(items[0] | {"spreadsheet_path": "spreadsheet/t1"},
                                    root / "sb400")
    gold = Path(instance.metadata["gold_path"]).read_bytes()
    work = tmp_path / "run" / "passes" / "data" / "t1_0"
    work.mkdir(parents=True)
    task.setup(instance, work)
    check_laid_item_files(task, work)
    laid = [p for p in work.rglob("*") if p.is_file()]
    assert [p.relative_to(work).as_posix() for p in laid] == ["item/input.xlsx"]
    assert all(p.read_bytes() != gold for p in laid)
    ws = workspace.Workspace(node="look", kind="scratch", of="", collect=(), root=tmp_path / "ws",
                             path=tmp_path / "ws" / "w")
    ws.path.mkdir(parents=True)
    with pytest.raises(workspace.WorkspaceError, match="not a file of the run directory"):
        workspace._seal(ws, work, ("bash",), (instance.metadata["gold_path"],))  # noqa: SLF001
    with pytest.raises(workspace.WorkspaceError, match="not a file of the run directory"):
        workspace._seal(ws, work, ("bash",), ("../../../../dataset/sb400/spreadsheet/t1/1_t1_golden.xlsx",))  # noqa: SLF001


def test_a_data_node_declares_its_tasks_item_files():
    """[validator, both runtimes refuse the state] V12: the Task's item files are the data
    node's writes; a Task declaring a grader file as one is refused there too."""
    issues = [str(i) for i in _issues(ifo.UndeclaredWrite())]
    assert any(i.startswith("V12 [data]") and "item file 'item/input.bin'" in i
               for i in issues), issues
    assert not [i for i in _issues(ifo.ItemFilesMode()) if i.code == "V12"]
    assert not [i for i in _issues(sb_mode.Solver()) if i.code == "V12"]


def _issues(mode):
    from refactory_dsh.nn.validate import validate

    state = mode.state_dict()
    graph = Graph.from_state(state["graph"])
    values = {k: v["value"] for k, v in state["knobs"].items()}
    return validate(graph, values, declared_knobs=mode.declare())


#: A stand-in DSH CLI that runs its probes sealed (`test_sealed_tools.AGENT`), reading the
#: item file where the step's workspace holds it.
def _sealed_cli(tmp_path: Path) -> Path:
    from test_sealed_tools import AGENT

    outside = tmp_path / "data"
    outside.mkdir()
    (outside / "gold.txt").write_text(ifo.gold("a"))
    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + AGENT.replace("cat input.txt", f"cat {ifo.INPUT}")
                      .replace("{grader}", str(outside / "gold.txt"))
                      .replace("{other}", str(tmp_path / "walker" / "facts.json")))
    script.chmod(0o755)
    return script


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_sealed_step_reads_the_item_file_and_nothing_golden(tmp_path, runtime):
    """[walker; spine, its agent dispatched through `worker_node`] A sealed step that lists
    the item file in its reads finds it in its workspace; the answer kept outside the pass
    and the run's files are denied to it."""
    from test_sealed_tools import _sealable

    from refactory_dsh.model import DshModelClient

    _sealable()
    client = DshModelClient(worker_backend="dsh", dsh_cli=str(_sealed_cli(tmp_path)),
                            attempts=1)

    def dispatched(state):
        return lambda wf: workers.lower_workflow(wf, tmp_path / "specs", client.context(),
                                                 lowering.node_declarations(state))

    records, ok = _records(ifo.SealedLook, runtime, tmp_path, model=client,
                           transform=dispatched if runtime == "spine" else None)
    assert ok, records
    assert sorted((r["id"], r["score"]) for r in records) == [("a", 1.0), ("b", 1.0)], records
    seen = [json.loads(path.read_text())
            for path in sorted(tmp_path.glob(f"{runtime}/passes/*/*/{ifo.ANSWER}"))]
    assert len(seen) == 2
    assert {s["input"] for s in seen} == {"INPUT-a", "INPUT-b"}
    assert all(s["grader"] == "DENIED" and s["other"] == "DENIED" for s in seen), seen


#: A runner's scope prefix whose user bus has gone away since it was made (systemd's user
#: manager stopped: its runtime directory goes with the person's last session, which on
#: some machines comes and goes every few seconds).
GONE_SCOPE = ("/usr/bin/env", "XDG_RUNTIME_DIR=/run/user/gone-for-the-test",
              "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/gone-for-the-test/bus",
              "/usr/bin/systemd-run", "--user", "--scope", "--quiet", "--collect", "--")

#: A stand-in DSH CLI that runs one command through the step's runner, as DSH's bash tool
#: would, and answers what it printed (`DENIED` when it printed nothing).
ONE_COMMAND = """\
import json, pathlib, subprocess, sys
cwd = pathlib.Path.cwd()
last = max(i for i, arg in enumerate(sys.argv) if arg == '--patch')
patch = json.loads(pathlib.Path(sys.argv[last + 1]).read_text())
runner = next(row['config']['runnerCommand'] for row in patch if row.get('id') == 'sandbox')
done = subprocess.run([*runner, '--ro-bind', '/', '/', '--', 'bash', '-c', 'cat {file}'],
                      cwd=cwd, capture_output=True, text=True, timeout=60)
(cwd / 'answer.txt').write_text(done.stdout.strip() or 'DENIED')
"""


def _one_command_client(tmp_path: Path, file: str):
    from refactory_dsh.model import DshModelClient

    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + ONE_COMMAND.replace("{file}", file))
    script.chmod(0o755)
    return DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)


def _spine_dispatch(tmp_path: Path, client) -> tuple[Path, dict]:
    """A sealed step reading ``input.txt``, as the spine dispatches it (`worker_node`)."""
    import shlex

    run = tmp_path / "run"
    run.mkdir()
    (run / "input.txt").write_text("THE-INPUT")
    graph = Graph()
    graph.add(nn_node("look", "agent", prompt_template="Look.", reads=("input.txt",),
                      writes=(ifo.ANSWER,), tools=("bash",), workspace="scratch"))
    state = {"graph": graph.to_state(), "knobs": {}}
    lowered = workers.lower_workflow(lowering.to_workflow(state), run, client.context(),
                                     lowering.node_declarations(state))
    return run, json.loads(Path(shlex.split(lowered.nodes["look"].dispatch_command)[-1])
                           .read_text())


LINUX_ONLY = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="systemd's user scope and unshare are Linux's")


@LINUX_ONLY
@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_scope_bus_gone_mid_step_runs_the_command_unscoped(tmp_path, runtime, monkeypatch):
    """[walker: a data pass in process; spine: its agent dispatch, `worker_node`] The
    step's runner was made with a systemd scope whose bus has since gone: each command
    runs sealed under the unscoped per-process limits, as a step started a moment later
    would, and the step reads its input -- never a command that silently did nothing."""
    from test_sealed_tools import _sealable

    from refactory_dsh import seal

    _sealable()
    monkeypatch.setattr(seal, "linux_scope", lambda: GONE_SCOPE)
    if runtime == "walker":
        client = _one_command_client(tmp_path, ifo.INPUT)
        records, ok = _records(ifo.SealedLook, "walker", tmp_path, model=client)
        assert ok and sorted((r["id"], r["score"]) for r in records) == [("a", 1.0), ("b", 1.0)], \
            records
        return
    from refactory_dsh import worker_node

    run, spec = _spine_dispatch(tmp_path, _one_command_client(tmp_path, "input.txt"))
    worker_node.execute(spec, run)
    assert (run / ifo.ANSWER).read_text() == "THE-INPUT"


@LINUX_ONLY
@pytest.mark.parametrize("unshare", ["missing", "exit 1", "exit 97"])
@pytest.mark.parametrize("runtime", ["walker", "spine"])
def test_a_seal_that_cannot_start_a_command_is_a_harness_incident(tmp_path, runtime, unshare,
                                                                  monkeypatch):
    """[walker: a data pass in process; spine: its agent dispatch, `worker_node`] The seal
    fails before the command starts -- its sandbox cannot be started, or its unshare (or
    systemd-run, or its own setup: "cannot bind /etc/group") exits 1 or 97 -- so no
    command runs, and the step is a harness incident naming why, never an item that
    quietly scores 0."""
    from test_sealed_tools import _sealable

    from refactory_dsh import items as items_mod
    from refactory_dsh import seal

    _sealable()
    monkeypatch.setattr(seal, "linux_scope", lambda: ())
    fake = tmp_path / "fake-unshare"
    if unshare != "missing":
        fake.write_text(f"#!/bin/sh\n{unshare}\n")
        fake.chmod(0o755)
    real_which = seal.shutil.which
    monkeypatch.setattr(seal.shutil, "which", lambda name, *a, **k: (
        str(fake) if name == "unshare" else real_which(name, *a, **k)))
    says = "SealFailed: harness incident: agent node 'look': "
    why = "before the command started"
    if runtime == "walker":
        client = _one_command_client(tmp_path, ifo.INPUT)
        records, ok = _records(ifo.SealedLook, "walker", tmp_path, model=client)
        assert not ok
        assert [r["error_kind"] for r in records] == ["harness_incident"] * 2, records
        assert all(says in str(r["error_text"]) and why in str(r["error_text"])
                   for r in records), records
        return
    from refactory_dsh import worker_node

    run, spec = _spine_dispatch(tmp_path, _one_command_client(tmp_path, "input.txt"))
    with pytest.raises(Exception) as raised:
        worker_node.execute(spec, run)
    text = f"look: {type(raised.value).__name__}: {raised.value}"
    assert says in text and why in text, text
    assert items_mod.error_kind(RuntimeError(text)) == "harness_incident"


@LINUX_ONLY
def test_a_command_that_started_never_makes_an_incident_whatever_it_exits(tmp_path):
    """[the runner both runtimes put in front of a sealed command] Once the seal says the
    command starts, its exit is its own: a command exiting 97 is no harness incident."""
    from test_sealed_tools import _layout, _sealable, _sealed

    from refactory_dsh import seal_reaper, workspace

    _sealable()
    run, place = _layout(tmp_path)
    done = _sealed(place, "echo ran; exit 97")
    assert done.returncode == 97 and done.stdout.split() == ["ran"], done
    assert not Path(seal_reaper.incidents_path(str(place.path))).exists()
    workspace.collect(place, run, [])


@LINUX_ONLY
def test_an_incident_belongs_to_the_one_dispatch_that_recorded_it(tmp_path):
    """[the seal's preparation and collection, both runtimes] A dispatch whose command could
    not start raises; the next dispatch of the step (its workspace reuses the name) does
    not, whether or not the first was collected."""
    from test_sealed_tools import _sealable

    from refactory_dsh import seal_reaper, workspace
    from refactory_dsh.incidents import SealFailed

    _sealable()
    run = tmp_path / "run"
    run.mkdir()
    (run / "input.txt").write_text("IN")

    def dispatch():
        return workspace.prepare(run, "look", "scratch", route={"worker_backend": "dsh"},
                                 tools=("bash",), reads=("input.txt",))

    first = dispatch()
    recorded = Path(seal_reaper.incidents_path(str(first.path)))
    seal_reaper.record_incident(str(first.path), "bus gone (simulated)")
    with pytest.raises(SealFailed, match="bus gone"):
        workspace.collect(first, run, [])
    assert not recorded.exists(), "consumed by the dispatch it belongs to"
    second = dispatch()
    assert second.path.name == "scratch-1"  # the name the first had
    workspace.collect(second, run, [])
    # One an earlier dispatch recorded and nothing collected (its session crashed), under
    # the name the next dispatch takes: that dispatch starts without it.
    seal_reaper.record_incident(str(second.root / "scratch-1"), "bus gone (simulated)")
    third = dispatch()
    assert third.path.name == "scratch-1"
    workspace.collect(third, run, [])


def test_the_report_counts_uncapped_steps_wherever_a_candidate_ran(tmp_path):
    """[the run's report] Every dispatch record under a candidate's run directory counts:
    its passes, its occasions (a fresh-evidence rescore, val, a smoke run) and theirs,
    once each even where a concurrent pass is also reached through its alias."""
    from refactory_dsh import loop as loop_mod

    runs = tmp_path / "runs"
    places = ["c1/passes/data/a_0", "c1/occasions/train_epoch_1/passes/data/a_0",
              "c1/occasions/val_1/passes/data/b_1", "c2/occasions/smoke_1", "c2"]
    for index, place in enumerate(places):
        record = runs / place / ".refactory" / "workspaces" / "look" / "dispatch.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"sealed": True, "whole_step_cap": index == len(places) - 1,
                                      "cap": f"no cap {index}"}))
    alias = runs / "c1" / ".refactory" / "live" / "data" / "a_0"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(runs / "c1" / "passes" / "data" / "a_0")
    found = loop_mod.uncapped_sealed_steps(runs)
    assert found["dispatches"] == 4, found


@LINUX_ONLY
def test_a_sealed_step_says_whether_it_had_a_whole_step_cap(tmp_path, monkeypatch):
    """[the step's dispatch record, both runtimes; the run's report] Without a scope, the
    record says the step has no whole-step cap and why; a runner that fell back mid-step
    says so too; the run's report says it once."""
    from test_sealed_tools import _sealable

    from refactory_dsh import loop as loop_mod
    from refactory_dsh import seal, seal_reaper, workspace

    _sealable()
    run = tmp_path / "runs" / "c1"
    run.mkdir(parents=True)
    (run / "input.txt").write_text("IN")
    monkeypatch.setattr(seal, "linux_scope", lambda: ())
    place = workspace.prepare(run, "look", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("input.txt",))
    record = json.loads((place.root / "dispatch.json").read_text())
    assert record["whole_step_cap"] is False and "per-process limits only" in record["cap"]
    monkeypatch.setattr(seal, "linux_scope", lambda: GONE_SCOPE)
    place = workspace.prepare(run, "look", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("input.txt",))
    assert json.loads((place.root / "dispatch.json").read_text())["whole_step_cap"] is True
    seal_reaper.record_note(str(place.path), "the bus went away (simulated)")
    workspace.collect(place, run, [])
    record = json.loads((place.root / "dispatch.json").read_text())
    assert record["whole_step_cap"] is False and "went away" in record["cap"]
    found = loop_mod.uncapped_sealed_steps(tmp_path / "runs")
    assert found == {"dispatches": 1, "why": "the bus went away (simulated)"}
    assert loop_mod._uncapped_lines(found)[1].startswith(  # noqa: SLF001
        "Sealed steps ran without a whole-step memory cap (1 step(s)): the bus went away")


def test_a_scope_is_made_only_where_the_user_manager_outlives_the_process(tmp_path, monkeypatch):
    """[the runner both runtimes make] A process outside the user's sessions and manager,
    for a user who does not linger, gets no scope: the manager would stop with the last
    session and take the scope's commands with it (killed mid-setup: "Terminated ...
    cannot bind /etc/group"), so its commands run unscoped from the start."""
    import pwd

    from refactory_dsh import seal

    uid = os.getuid()
    name = pwd.getpwuid(uid).pw_name
    service = "0::/system.slice/some-runner.service\n"
    assert not seal.user_manager_kept(service, str(tmp_path))
    assert seal.user_manager_kept(f"0::/user.slice/user-{uid}.slice/session-3.scope\n",
                                  str(tmp_path))
    assert seal.user_manager_kept(
        f"0::/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/run.scope\n", str(tmp_path))
    (tmp_path / name).write_text("")
    assert seal.user_manager_kept(service, str(tmp_path))
    monkeypatch.setattr(seal, "user_manager_kept", lambda *a, **k: False)
    monkeypatch.setattr(seal.shutil, "which", lambda name, *a, **k: "/usr/bin/systemd-run")
    monkeypatch.setattr(seal.os.path, "isdir", lambda path: True)
    assert seal._scope_prefix() == []  # noqa: SLF001


def test_a_kept_scope_whose_bus_is_gone_is_probed_again(monkeypatch):
    """[the runner both runtimes make] A scope found earlier whose bus has gone is not
    used again: later commands get a fresh probe's answer (unscoped where there is none)."""
    from refactory_dsh import seal

    monkeypatch.setattr(seal, "_SCOPE", [GONE_SCOPE])
    monkeypatch.setattr(seal, "_scope_prefix", lambda: [])
    assert seal.linux_scope() == ()
    assert seal._SCOPE == []  # noqa: SLF001


@LINUX_ONLY
def test_the_runner_falls_back_unscoped_or_records_why_a_command_could_not_start(tmp_path):
    """[the runner both runtimes put in front of a sealed command] With its scope's bus
    gone, a runner of the seal's own shape runs unscoped under the tight limits; one it
    cannot rewrite is refused (exit 97, the command never runs) and recorded beside the
    workspace for `workspace.collect`."""
    import subprocess as sp

    from refactory_dsh import seal_reaper

    scoped = ["unshare", "-Urnmp", "--", "bash", "script", "refactory-seal", "/w", "1", "1",
              f"{seal_reaper.LIMITS} {seal_reaper.LINUX_LIMITS}", "close"]
    assert seal_reaper.unscoped([*GONE_SCOPE, *scoped]) == [
        *scoped[:9], f"{seal_reaper.LIMITS} {seal_reaper.LINUX_LIMITS_UNSCOPED}", "close"]
    ws = tmp_path / "scratch-1"
    ws.mkdir()
    marker = tmp_path / "ran"
    command = [*GONE_SCOPE, "/bin/sh", "-c", f"touch {marker}", "refactory-seal", str(ws)]
    done = sp.run([sys.executable, "-I", seal_reaper.__file__, "--exec", *command],
                  capture_output=True, text=True, timeout=60)
    assert done.returncode == 97 and "user scope bus" in done.stderr
    assert not marker.exists()
    recorded = Path(seal_reaper.incidents_path(str(ws))).read_text().splitlines()
    assert len(recorded) == 1 and "is gone" in json.loads(recorded[0])["reason"]


# ── B. 'to' an agent whose prompt is no knob ──────────────────────────────


def _sb_attempt(edits: list[dict], **settings):
    mode = sb_mode.Solver()
    optimizer = Optimizer()
    optimizer.sealed_tools = (True, "")
    for name, value in settings.items():
        setattr(optimizer, name, value)
    seed = Candidate.from_state(mode.state_dict())
    return optimizer.attempt(Proposal(edits=edits), seed, mode.invariants(), True, (),
                             protected=mode.edit_protected(), record=False)


INSPECT = {"op": "insert", "after": "frame", "id": "inspect", "kind": "agent",
           "text": "Describe the workbook's layout.", "tools": ["bash"],
           "sees": [sb_tasks.INPUT]}


def test_one_insert_reaches_the_solver_on_spreadsheetbench():
    """[edit layer; the state is held to both runtimes in the shapes below] The probe's
    edit, one line: a sealed step seeing the item's input, its output put in solve's
    prompt (a template, no knob). solve's frozen system prompt stays as it was."""
    out = _sb_attempt([{**INSPECT, "to": "solve"}])
    assert isinstance(out, Candidate), out
    nodes = out.state["graph"]["nodes"]
    assert nodes["inspect"]["reads"] == [sb_tasks.INPUT]
    assert nodes["inspect"]["tools"] == ["bash"]
    task = out.state["knobs"]["solve__task"]["value"]
    assert task.startswith("{{solver_prompt}}") and "{{inspect}}" in task
    seed = sb_mode.Solver().state_dict()
    assert out.state["knobs"]["solve__system_prompt"] == seed["knobs"]["solve__system_prompt"]
    assert nodes["solve"]["tool_policy"] == "refused"


@pytest.mark.parametrize("to", ["solve__system_prompt", "skill", "max_completion_tokens"])
def test_to_a_frozen_or_unfilled_knob_is_still_refused(to):
    """[edit layer] What truly cannot take the output stays refused."""
    out = _sb_attempt([{**INSPECT, "to": to}])
    assert isinstance(out, list) and any("E-wiring" in r for r in out), out


def test_without_to_the_new_step_is_unread_and_the_hint_names_the_agent():
    """[edit layer] With no 'to', nothing reads the new step; the refusal says to name the
    agent in 'to'."""
    out = _sb_attempt([INSPECT])
    assert isinstance(out, list) and out[0].startswith("E-unread"), out
    assert "name solve in 'to'" in out[0], out


@pytest.mark.parametrize("seed,edit,target", [
    ("fuzz_fork_join", {"op": "insert", "before": "left", "kind": "agent", "id": "note",
                        "text": "Note what matters."}, "left"),
    ("fuzz_nested_fork", {"op": "insert", "before": "deep_b", "kind": "agent", "id": "note",
                          "text": "Note what matters."}, "deep_b"),
    ("fuzz_gate_loop", {"op": "insert", "before": "work", "kind": "agent", "id": "note",
                        "text": "Note what matters."}, "work"),
    ("fuzz_data_pass", {"op": "insert", "after": "read_item", "kind": "agent", "id": "note",
                        "text": "Note what matters."}, "solve"),
], ids=["fork branch", "nested fork", "loop", "data pass"])
@needs_spine
def test_to_an_agent_whose_prompt_is_no_knob_reslots_it_in_every_shape(seed, edit, target):
    """[walker and spine: `structure_fuzz.check_state`] In a fork branch, a nested fork, a
    loop and a data pass, 'to' naming an agent whose prompt is a template gives it the new
    step's section; its id, place and outputs stay."""
    applier = sf.Applier(sf.seeds()[seed])
    parent = Candidate.from_state(applier.seed_state)
    before = Graph.from_state(applier.seed_state["graph"]).nodes[target]
    assert core_ops._task_knob(before) is None  # noqa: SLF001
    out = applier.attempt(parent, {**edit, "to": target}, {parent.id})
    assert isinstance(out, Candidate), out
    after = Graph.from_state(out.state["graph"]).nodes[target]
    knob = core_ops._task_knob(after)  # noqa: SLF001
    assert knob == f"{target}__task"
    text = out.state["knobs"][knob]["value"]
    assert text.startswith(before.prompt_template.rstrip()) and "{{note}}" in text
    assert (after.writes, after.produces) == (before.writes, before.produces)
    assert sf.check_state(applier, out.state, walker=True, spine=True) == []


def test_a_state_this_version_saved_is_never_upgraded():
    """[load, both runtimes' entry] An edit's node calling an op that opens the input, with
    reads of its own choosing, in a state saved by this version: returned as it is, with
    no note and no warning (its data node already declares the input)."""
    import warnings

    state = sb_mode.Solver().state_dict()
    nodes = state["graph"]["nodes"]
    nodes["run_turn_fallback"] = {**copy.deepcopy(nodes["run_turn"]), "reads": ["reply.txt"]}
    if "id" in nodes["run_turn_fallback"]:
        nodes["run_turn_fallback"]["id"] = "run_turn_fallback"
    before = json.dumps(state, sort_keys=True)
    out, notes = sb_mode.Solver.upgrade_state(state)
    assert notes == [] and json.dumps(out, sort_keys=True) == before
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mode = sb_mode.Solver(state=state)
    assert mode.state_upgrades == ()
    assert mode.state_dict()["graph"]["nodes"]["run_turn_fallback"]["reads"] == ["reply.txt"]


# ── C. the request ────────────────────────────────────────────────────────


def test_the_operator_table_offers_no_tools_when_the_run_has_none():
    """[request text, the same tools answer as the operator lines] SpreadsheetBench's table:
    with no tool step, no refusal says 'tools'; with one, its refusals are as before."""
    from test_operator_fit import _section

    mode = sb_mode.Solver()
    state, protected = mode.state_dict(), tuple(mode.edit_protected())
    on = "\n".join(_section(state, protected=protected, limit=10 ** 6, tools=True))
    off = "\n".join(_section(state, protected=protected, limit=10 ** 6, tools=False))
    assert "'tools'" not in off
    assert "give it a code step, or 'tools'" in on
    assert off == core_ops.refusal_without_tools(on)


def test_a_run_that_offers_no_tool_step_offers_no_tools_anywhere():
    """[request text] With no tool step (no seal, or ``new_step_tools`` off) no operator
    line or field offers 'tools'; with one, insert's and map's do."""
    from refactory_dsh.optim.memory import state_view

    view = state_view(sb_mode.Solver().state_dict())
    off = "\n".join(base_mod.operation_lines(nn_edits.catalogue("broad"), view.node_states,
                                             view.edges, tools=False))
    on = "\n".join(base_mod.operation_lines(nn_edits.catalogue("broad"), view.node_states,
                                            view.edges, tools=True))
    core = [line for line in off.splitlines() if line.startswith("- `")]
    assert core and not [line for line in core if "tools" in line], core
    assert core_ops.INSERT_TOOLS in on and core_ops.MAP_TOOLS in on
    assert base_mod.NO_TOOLS_LINE in _sections(tools=False)


def test_the_request_says_new_steps_may_have_tools_whatever_the_existing_steps_rules():
    """[request text] An existing step keeps its tool rules; a new step may have tools,
    sealed; said without any domain's words."""
    line = core_ops.OPERATORS["insert"]["line"]
    assert "an existing step keeps its own tool rules" in line
    assert "no grader files" in line
    assert "an item's files too" in line
    assert "existing step keeps its own tool rules" in base_mod.TOOLS_LINE
    assert "sealed" in base_mod.TOOLS_LINE and "no network" in base_mod.TOOLS_LINE
    for text in (line, base_mod.TOOLS_LINE, base_mod.NO_TOOLS_LINE, base_mod.PIPELINE_VIEW_LEAD):
        assert not {"spreadsheet", "xlsx", "workbook", "solve"} & set(
            w.strip(".,;:()`'\"").lower() for w in text.split()), text
    view = _sections(tools=True)
    assert base_mod.TOOLS_LINE in view and base_mod.NO_TOOLS_LINE not in view
    view = _sections(tools=False)
    assert base_mod.NO_TOOLS_LINE in view and base_mod.TOOLS_LINE not in view


def _sections(*, tools: bool) -> list[str]:
    from refactory_dsh.optim.memory import state_view

    view = state_view(sb_mode.Solver().state_dict())
    return base_mod.composition_section(nn_edits.catalogue("broad"), view.node_states,
                                        view.edges, tools=tools)


def test_the_pipeline_view_shows_the_item_files_and_who_reads_them():
    """[request text] The per-item line names each item file and its readers, a new step
    that sees one among them."""
    protected = sb_mode.Solver().edit_protected()
    first = pv.render(sb_mode.Solver().state_dict(), protected).splitlines()[0]
    assert first == ("per item (data -> current_item.json, item/input.xlsx; item files: "
                     "item/input.xlsx seen by finalize, prepare, run_turn) {")
    out = _sb_attempt([{**INSPECT, "to": "solve"}])
    assert "item/input.xlsx seen by finalize, inspect, prepare, run_turn" in \
        pv.render(out.state, protected).splitlines()[0]


def _table_rows(state: dict, protected: tuple) -> dict[str, list[str]]:
    from test_operator_fit import _rows, _section

    return _rows(_section(state, protected=protected, limit=10 ** 6))


#: The most a step's "carries" cell may hold in a loop.
CARRIES_CAP = 600


def test_the_operator_table_stays_short_in_a_loop_and_agrees_with_an_attempt():
    """[request text, checked by attempts] In SpreadsheetBench's turn loop each producer is
    followed once and what feeds a step from the loop's previous round is said so; the
    rows' verdicts are an attempt's, and B's 'to' shows inserts before solve applying."""
    from test_operator_fit import _check_rows_are_true

    mode = sb_mode.Solver()
    state, protected = mode.state_dict(), tuple(mode.edit_protected())
    rows = _check_rows_are_true(state, protected)
    carries = rows["solve"][0]
    assert len(carries) <= CARRIES_CAP, carries
    assert "op run_turn, from the loop's previous round" in carries
    assert carries.count("op prepare, which reads") == 1
    assert "+ to solve: insert" in rows["frame"][1], rows["frame"]
    assert "rewire (sees task alone)" in rows["solve"][2]


@pytest.mark.parametrize("seed", ["fuzz_gate_loop", "fuzz_nested_fork", "fuzz_data_pass"])
def test_every_carries_cell_is_bounded_in_other_shapes(seed):
    """[request text] Loops, nested forks and data passes keep each cell short."""
    applier = sf.Applier(sf.seeds()[seed])
    rows = _table_rows(applier.seed_state, applier.protected)
    assert rows and all(len(cells[0]) <= CARRIES_CAP for cells in rows.values()), rows


# ── E. new_step_tools ─────────────────────────────────────────────────────


def test_new_step_tools_off_refuses_every_new_step_with_tools():
    """[attempt path] Off, a sealed new step is refused naming the setting; on (the
    default) the same edit is taken."""
    off = _sb_attempt([{**INSPECT, "to": "solve"}], new_step_tools=False)
    assert isinstance(off, list) and off[0].startswith("E-new-node-tools"), off
    assert "new_step_tools: false" in off[0]
    assert isinstance(_sb_attempt([{**INSPECT, "to": "solve"}]), Candidate)


def test_new_step_tools_off_refuses_a_copy_of_a_seed_tool_step_too():
    """[attempt path] A copy of a seed step that has tools is a new step with tools."""
    seed = {"graph": {"nodes": {"a": {"kind": "agent", "tool_policy": None, "tools": ["bash"]}}}}
    copy_state = {"graph": {"nodes": {**seed["graph"]["nodes"], "a_copy1": {
        "kind": "agent", "tool_policy": None, "tools": ["bash"],
        "workspace": {"kind": "scratch"}}}}}
    assert base_mod._new_node_tools(seed, copy_state, (True, "")) == []  # noqa: SLF001
    refused = base_mod._new_node_tools(seed, copy_state, (True, ""), allowed=False)  # noqa: SLF001
    assert refused and "new_step_tools: false" in refused[0]


def test_new_step_tools_is_a_profile_switch_on_by_default(tmp_path):
    """[settings] On by default; off offers no tool step; any other value is refused; the
    default leaves the resolved digest as it was."""
    from refactory_dsh import profiles
    from refactory_dsh.loop import Loop, LoopConfig

    assert LoopConfig().new_step_tools is True
    assert profiles.defaults()["new_step_tools"] is True
    assert profiles.normalize({**profiles.defaults(), "new_step_tools": 0})["new_step_tools"] is False
    with pytest.raises(profiles.ProfileError, match="new_step_tools must be true or false"):
        profiles.normalize({**profiles.defaults(), "new_step_tools": "off"})
    loop = Loop("livemath/skill_graph",
                LoopConfig(runs_root=tmp_path / "runs", new_step_tools=False, optimizer="random"),
                project=Path("."))
    assert loop._tools_offered() is False  # noqa: SLF001
    assert loop._tools_not_offered() == base_mod.NEW_STEP_TOOLS_OFF  # noqa: SLF001
    assert loop.optimizer.new_step_tools is False
