"""A tool step the outer loop adds in the per-item part works on the item's input files,
and runs its commands with the Task's declared tool environment.

- **Item inputs by default, never forced** (`core_ops._with_item_files`,
  `core_ops.default_item_inputs`): a new sealed tool step in a per-item pass whose inputs
  the proposer did not name (no 'sees' or 'reads') gets the item's input files
  (`Task.item_files`) in its declared ``reads``; one whose inputs it named keeps exactly
  those, and the candidate's notes say whether each such step sees the item's inputs.
  Outside a pass, or without tools, nothing is added.
- **Wiring notes judged on the candidate**: what is true of the candidate (a new step's
  output nothing reads) is said from the state after every edit of the proposal.
- **The tool environment** (`Task.tool_environment`, `seal.tool_path`): a sealed step's
  ``python3``/``python`` is the Task's interpreter, on the walker and on the TS spine
  (its agent dispatched through `worker_node`); a Task that declares none runs as before;
  one that cannot be had is refused.
- **The request** states both as facts where tools are offered, and nothing of them
  where they are not (``new_step_tools`` off).

Each test names the runtimes it covers in its docstring.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import item_files_ops as ifo
from refactory_dsh import lowering, seal, workers
from refactory_dsh.nn import core_ops
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.optim import base as base_mod
from refactory_dsh.task import task_tool_environment
from refactory_dsh.workspace import is_sealed
from refactory_packages.spreadsheetbench import interpreter as sb_interpreter
from refactory_packages.spreadsheetbench import mode as sb_mode
from refactory_packages.spreadsheetbench import tasks as sb_tasks
from test_structure_reach import RUNTIMES, _records, _sb_attempt


def _attempt(mode, edits: list[dict], **settings):
    optimizer = Optimizer()
    optimizer.sealed_tools = (True, "")
    for name, value in settings.items():
        setattr(optimizer, name, value)
    seed = Candidate.from_state(mode.state_dict())
    return optimizer.attempt(Proposal(edits=edits), seed, mode.invariants(), True, (),
                             protected=mode.edit_protected(), record=False)


def _nodes(out) -> dict:
    assert isinstance(out, Candidate), out
    return out.state["graph"]["nodes"]


LOOK = {"op": "insert", "before": "solve", "id": "look", "kind": "agent",
        "text": "Look at the item.", "tools": ["bash"], "to": "solve"}


# ── item inputs by default ────────────────────────────────────────────────


def test_a_tool_step_an_operator_adds_in_the_pass_is_given_the_item_inputs():
    """[edit layer, any Task] insert and fallback: the new sealed step's reads hold the
    item's input though the proposal never named it, and the candidate says so."""
    mode = ifo.AgentPass()
    out = _attempt(mode, [LOOK])
    nodes = _nodes(out)
    assert nodes["look"]["reads"] == [ifo.INPUT]
    assert nodes["look"]["tools"] == ["bash"] and is_sealed(nodes["look"])
    assert f"look (tools, in the per-item part) has the item's input file(s) {ifo.INPUT} " \
        "in its sealed workspace" in out.notes, out.notes
    out = _attempt(mode, [{"op": "fallback", "step": "solve", "kind": "agent",
                           "tools": ["bash"]}])
    assert ifo.INPUT in _nodes(out)["solve_fallback"]["reads"]


def test_a_tool_step_that_names_the_input_keeps_it_once():
    """[edit layer] What the step names stays, the input once."""
    out = _attempt(ifo.AgentPass(), [{**LOOK, "sees": [ifo.INPUT]}])
    assert _nodes(out)["look"]["reads"] == [ifo.INPUT]


def test_a_step_whose_sees_leaves_the_input_out_is_accepted_without_it():
    """[edit layer] A default, not a limit: 'sees' that names other inputs (or none) is
    kept as written, and the candidate says the step does not see the item's input."""
    for sees in ([], ["current_item.json"]):
        out = _attempt(ifo.AgentPass(), [{**LOOK, "sees": sees}])
        assert ifo.INPUT not in (_nodes(out)["look"].get("reads") or []), sees
        assert f"look (tools, in the per-item part) does not see the item's input file(s) " \
            f"{ifo.INPUT}" in out.notes, out.notes


def _add_node(**node) -> dict:
    return {"op": "add_node", "position": {"before": "solve"},
            "node": {"id": "look", "kind": "agent", "prompt_template": "Look at the item.",
                     "writes": ["look.txt"], "tools": ["bash"], "tool_policy": None,
                     "workspace": {"kind": "scratch"}, **node}}


def test_a_written_out_tool_node_takes_the_same_default():
    """[edit layer] add_node with tools in the pass and no 'reads' gets the item's input,
    recorded in the edit the candidate keeps (so it replays); 'reads' given stays."""
    mode = ifo.AgentPass()
    out = _attempt(mode, [_add_node()])
    nodes = _nodes(out)
    assert nodes["look"]["reads"] == [ifo.INPUT]
    assert out.edits[-1]["node"]["reads"] == [ifo.INPUT]
    assert nn_edits.apply_many(out.edits, mode.state_dict())["graph"]["nodes"]["look"][
        "reads"] == [ifo.INPUT]
    kept = _attempt(mode, [_add_node(reads=[])])
    assert _nodes(kept)["look"].get("reads", []) == []
    assert any("look (tools, in the per-item part) does not see" in note
               for note in kept.notes), kept.notes


@pytest.mark.parametrize("edit", [
    {**LOOK, "before": "report", "id": "top", "to": "report"},
    {k: v for k, v in LOOK.items() if k != "tools"},
], ids=["outside the pass", "no tools"])
def test_outside_the_pass_or_without_tools_no_item_file_is_added(edit):
    """[edit layer] A tool step after the pass (no item there) and a step without tools
    get nothing they did not name."""
    nodes = _nodes(_attempt(ifo.AgentPass(), [edit]))
    assert nodes[edit["id"]].get("reads", []) == []
    assert not any(ifo.INPUT in (n.get("reads") or []) for n in nodes.values()
                   if n["kind"] == "op" and n.get("call", "").endswith(":collect"))


#: The shakedown's structural edit, as the proposer wrote it: a tool step before solve
#: that names no file, and solve rewired to see its output.
SHAKEDOWN = [{"op": "insert", "before": "solve", "id": "explore", "kind": "agent",
              "text": "Explore the input workbook and report its layout.", "tools": ["bash"]},
             {"op": "rewire", "step": "solve",
              "sees": ["skill_section", "solver_prompt", "explore"]}]


def test_the_shakedown_s_explore_step_gets_the_input_and_is_said_to_be_read():
    """[edit layer, SpreadsheetBench; the candidate passes lowering] The live edit: explore
    reads item/input.xlsx, and the candidate's notes say solve reads it (no "no step
    reads what explore makes")."""
    out = _sb_attempt(SHAKEDOWN)
    nodes = _nodes(out)
    assert nodes["explore"]["reads"] == [sb_tasks.INPUT]
    assert "{{explore}}" in out.state["knobs"]["solve__task"]["value"]
    assert not any("no step reads what explore" in note for note in out.notes), out.notes
    assert any("explore (tools, in the per-item part) has the item's input file(s) "
               "item/input.xlsx" in note for note in out.notes), out.notes


def test_a_truly_unread_new_step_is_still_noted():
    """[edit layer] With no later edit reading it, the note stays; judged on the state
    after the reading edit, it goes."""
    state = sb_mode.Solver().state_dict()
    after = nn_edits.apply(SHAKEDOWN[0], state)
    final = nn_edits.apply(SHAKEDOWN[1], after)
    alone = core_ops.wiring_notes(SHAKEDOWN[0], state, after)
    assert any("no step reads what explore make(s) yet" in note for note in alone), alone
    judged = core_ops.wiring_notes(SHAKEDOWN[0], state, after, final)
    assert not any("no step reads what explore" in note for note in judged), judged
    assert any("added explore" in note for note in judged), judged


def test_a_dead_knob_is_said_by_the_edit_that_killed_it_only():
    """[edit layer] The S4 bundle: the branch leaves hypotheses_task alive and the rewire
    kills it, so only the rewire's note says it (as on main)."""
    from test_core_op_fixes import BRANCH_REWIRE, INDEP, _accepted

    out = _accepted(INDEP, BRANCH_REWIRE)
    said = [note.split(":", 1)[0] for note in out.notes
            if "knob 'hypotheses_task' now reaches no step" in note]
    assert said == ["rewire"], out.notes


def test_a_rewired_tool_step_keeps_its_tools_and_exactly_what_it_says():
    """[edit layer] A rewire of a sealed step rebuilds it: it still runs sealed (its tool
    policy is not reset to a new step's 'none'), its files are its own reads (no collect
    step), and it sees exactly what the rewire says: the item's input is not put back."""
    mode = ifo.AgentPass()
    first = _attempt(mode, [LOOK])
    out = Optimizer()
    out.sealed_tools = (True, "")
    again = out.attempt(Proposal(edits=[{"op": "rewire", "step": "look",
                                         "sees": ["current_item.json"],
                                         "text": "Look again."}]),
                        first, mode.invariants(), True, (), protected=mode.edit_protected(),
                        record=False)
    look = _nodes(again)["look"]
    assert is_sealed(look) and look["tools"] == ["bash"], look
    assert look["reads"] == ["current_item.json"], look
    assert "look_inputs" not in _nodes(again)
    assert any("look (tools, in the per-item part) does not see" in note
               for note in again.notes), again.notes


# ── the request ───────────────────────────────────────────────────────────


def _operation_text(state: dict, *, tools: bool) -> str:
    from refactory_dsh.optim.memory import state_view

    view = state_view(state)
    return "\n".join(base_mod.operation_lines(nn_edits.catalogue("broad"), view.node_states,
                                              view.edges, tools=tools))


def test_the_request_states_the_tool_environment_only_where_tools_are_offered():
    """[request text: the proposer's and the analysts' operations, one source] With tool
    steps on, the Task's environment is said in its own words; off, nothing of it."""
    state = sb_mode.Solver().state_dict()
    on = _operation_text(state, tools=True)
    off = _operation_text(state, tools=False)
    line = ("- in the per-item part `data`, a step with `tools` runs its commands with "
            + sb_interpreter.SUMMARY)
    assert line in on.splitlines()
    assert sb_interpreter.SUMMARY not in off and "a step with `tools`" not in off
    assert "in the per-item part, without 'sees', the item's input files too" in on
    assert "item's input files" not in off
    # A Task that declares no environment: no such line, tools or not.
    plain = _operation_text(ifo.ItemFilesMode().state_dict(), tools=True)
    assert "runs its commands with" not in plain


def test_spreadsheetbench_declares_the_interpreter_its_model_code_runs_under():
    """[Task declaration] The same interpreter and check as generated code
    (`interpreter.interpreter`), said from the pins; core code names no dataset."""
    env = task_tool_environment(sb_mode.TASK)
    assert env is sb_tasks.SpreadsheetBenchTask.tool_environment
    assert env.python is sb_interpreter.interpreter
    assert env.summary == (f"python3 (and python) = Python {sb_interpreter.PYTHON} with "
                           f"openpyxl {sb_interpreter.PINS['openpyxl']}, pandas "
                           f"{sb_interpreter.PINS['pandas']}, numpy {sb_interpreter.PINS['numpy']}")
    for text in (core_ops.INSERT_TOOLS, base_mod.TOOLS_LINE):
        assert not {"spreadsheet", "xlsx", "openpyxl", "pandas"} & set(
            w.strip(".,;:()`'\"").lower() for w in text.split()), text


# ── the tool environment, on both runtimes ────────────────────────────────


#: A stand-in DSH CLI: runs its probes through the ``runnerCommand`` its patch gives DSH's
#: sandbox provider, as DSH's bash tool would, and writes what python3 and python are.
ENV_AGENT = """\
import json, pathlib, subprocess, sys
cwd = pathlib.Path.cwd()
last = max(i for i, arg in enumerate(sys.argv) if arg == '--patch')
patch = json.loads(pathlib.Path(sys.argv[last + 1]).read_text())
runner = next(row['config']['runnerCommand'] for row in patch if row.get('id') == 'sandbox')
def bash(script):
    done = subprocess.run([*runner, '--ro-bind', '/', '/', '--', 'bash', '-c', script],
                          cwd=cwd, capture_output=True, text=True, timeout=120)
    return done.stdout.strip() or 'DENIED: ' + done.stderr.strip()[-300:]
answer = {'input': bash('cat item/input.bin'),
          'python3': bash('python3 -c "import sys; print(sys.prefix)"'),
          'python': bash('python -c "import sys; print(sys.prefix)"'),
          'which': bash('command -v python3')}
bash('echo ' + json.dumps(json.dumps(answer)) + ' > answer.txt')
"""


def _env_client(tmp_path: Path):
    from refactory_dsh.model import DshModelClient

    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + ENV_AGENT)
    script.chmod(0o755)
    return DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)


def _run_sealed(mode_cls, runtime: str, tmp_path: Path):
    client = _env_client(tmp_path)

    def dispatched(state):
        return lambda wf: workers.lower_workflow(wf, tmp_path / "specs", client.context(),
                                                 lowering.node_declarations(state))

    return _records(mode_cls, runtime, tmp_path, model=client,
                    transform=dispatched if runtime == "spine" else None)


def _prefix(python: str) -> str:
    return subprocess.run([python, "-c", "import sys; print(sys.prefix)"], capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_sealed_step_runs_the_task_s_interpreter_as_python3(tmp_path, runtime):
    """[walker; spine, its agent dispatched through `worker_node`] python3 and python in a
    sealed step's commands are the Task's interpreter; its dispatch record says which."""
    from test_sealed_tools import _sealable

    _sealable()
    records, ok = _run_sealed(ifo.SealedEnvLook, runtime, tmp_path)
    assert ok, records
    assert sorted((r["id"], r["score"]) for r in records) == [("a", 1.0), ("b", 1.0)], records
    dispatches = sorted(tmp_path.glob(f"{runtime}/passes/*/*/.refactory/workspaces/look/"
                                      "dispatch.json"))
    assert len(dispatches) == 2
    for dispatch in dispatches:
        recorded = json.loads(dispatch.read_text())["tool_environment"]
        assert recorded["task"] == f"{ifo.MODULE}:ToolEnvTask"
        assert recorded["summary"] == ifo.TOOL_SUMMARY
        answer = json.loads((dispatch.parents[3] / ifo.ANSWER).read_text())
        expected = _prefix(recorded["python"])
        assert answer["python3"] == expected and answer["python"] == expected, answer
        assert answer["which"].endswith("/tool-bin/python3"), answer
        assert answer["input"].startswith("INPUT-"), answer


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_task_that_declares_no_environment_runs_the_machine_s_python(tmp_path, runtime):
    """[walker; spine] Without a declaration nothing changes: the machine's own python3, no
    tool environment in the dispatch record."""
    from test_sealed_tools import _sealable

    _sealable()
    records, ok = _run_sealed(ifo.SealedLook, runtime, tmp_path)
    assert ok, records
    for dispatch in sorted(tmp_path.glob(f"{runtime}/passes/*/*/.refactory/workspaces/look/"
                                         "dispatch.json")):
        assert "tool_environment" not in json.loads(dispatch.read_text())
        answer = json.loads((dispatch.parents[3] / ifo.ANSWER).read_text())
        assert "tool-bin" not in answer["which"], answer


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_tool_environment_that_cannot_be_had_fails_the_step(tmp_path, runtime):
    """[walker; spine] The Task's interpreter is missing: the step is refused with the
    Task's reason, never run with the machine's python instead."""
    from test_sealed_tools import _sealable

    _sealable()
    records, _ok = _run_sealed(ifo.SealedMissingEnv, runtime, tmp_path)
    assert records and all(r["score"] == 0.0 for r in records), records
    assert not list(tmp_path.glob(f"{runtime}/passes/*/*/{ifo.ANSWER}"))
    errors = json.dumps(records)
    assert "pinned interpreter is missing" in errors, errors


# ── what the seal binds for an interpreter ────────────────────────────────


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A run with item ``a``'s pass and a sealed workspace in it; the run's root."""
    run = tmp_path / "run"
    pass_dir = run / "passes" / "data" / "a"
    workspace = pass_dir / ".refactory" / "workspaces" / "look" / "scratch-1"
    workspace.mkdir(parents=True)
    return run, pass_dir, workspace


def _bound(monkeypatch, *folders: Path) -> None:
    """The interpreter reports running from ``folders`` (`seal.interpreter_dirs`)."""
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(seal, "interpreter_dirs", lambda _python: tuple(map(str, folders)))


@pytest.mark.parametrize("rule", ["holds the run", "inside the run", "holds home",
                                  "holds refactory's own", "inside refactory's own",
                                  "inside a Task's hidden path"])
def test_each_refusal_rule_for_an_interpreter_folder(tmp_path, monkeypatch, rule):
    """[seal] One rule each: a folder the interpreter runs from that would show the run,
    another item's pass, the home directory, refactory's own directory or a Task's
    grader-side path is refused (the workspace is in the run); a folder elsewhere is
    bound."""
    run, _pass, workspace = _layout(tmp_path)
    own = tmp_path / "refactory-home"
    monkeypatch.setenv("REFACTORY_HOME", str(own))
    own.mkdir()
    home = tmp_path / "home" / "person"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    folder = {"holds the run": tmp_path, "inside the run": run / "passes" / "data" / "b" / "venv",
              "holds home": tmp_path / "home", "holds refactory's own": own.parent,
              "inside refactory's own": own / "venv", "inside a Task's hidden path":
              dataset / "sb"}[rule]
    if folder in (tmp_path, own.parent):
        # Holding the run or refactory's own (not inside one): no other rule catches it.
        folder = tmp_path / "wide"
        run_parent = folder
        _bound(monkeypatch, folder)
        target = run_parent / ("run" if rule == "holds the run" else "own")
        target.mkdir()
        if rule == "holds the run":
            runs = [target]
        else:
            monkeypatch.setenv("REFACTORY_HOME", str(target))
            runs = [run]
    else:
        _bound(monkeypatch, folder)
        runs = [run]
    with pytest.raises(seal.SealError, match="may not read"):
        seal.tool_path(sys.executable, workspace.parent / "tool-bin", workspace=workspace,
                       runs=runs, hidden=[str(dataset)])
    _bound(monkeypatch, tmp_path / "elsewhere" / "venv")
    made = seal.tool_path(sys.executable, workspace.parent / "tool-bin", workspace=workspace,
                          runs=[run], hidden=[str(dataset)])
    assert sorted(p.name for p in Path(made.bin).iterdir()) == ["python", "python3"]
    assert made.bin in made.reads and str(tmp_path / "elsewhere" / "venv") in made.reads


def test_an_interpreter_named_through_a_link_binds_none_of_the_link_s_folders(tmp_path):
    """[seal, Linux, sealed command] The reviewer's probe: the interpreter named as
    ``<x>/dataset/sb/python``, a link beside the gold file. Only what the interpreter
    reports binds, so the gold file is not there."""
    from test_sealed_tools import _sealable

    _sealable()
    run, _pass, workspace = _layout(tmp_path)
    dataset = tmp_path / "x" / "dataset"
    (dataset / "sb").mkdir(parents=True)
    (dataset / "gold.txt").write_text("GOLD-SECRET")
    (dataset / "sb" / "python").symlink_to(sys.executable)
    seal.interpreter_dirs.cache_clear()
    made = seal.tool_path(str(dataset / "sb" / "python"), workspace.parent / "tool-bin",
                          workspace=workspace, runs=[run])
    assert not any(str(tmp_path / "x") in folder for folder in made.reads), made.reads
    result = _sealed(workspace, made, f"cat {dataset / 'gold.txt'}; python3 -c 'print(1)'")
    assert "GOLD-SECRET" not in result.stdout + result.stderr, result
    assert result.stdout.split() == ["1"], result


def _sealed(workspace: Path, made, script: str) -> subprocess.CompletedProcess:
    argv = [*seal.runner_command(workspace, seal.step_cap(workspace), made),
            "--ro-bind", "/", "/", "--", "bash", "-c", script]
    return subprocess.run(argv, cwd=workspace, capture_output=True, text=True, timeout=120)


def test_a_venv_in_another_item_s_pass_is_refused_through_the_workspace(tmp_path, monkeypatch):
    """[both runtimes' `workspace.prepare`] The pass's run root, not only the pass, is
    refused: an interpreter in item b's pass never serves item a's step."""
    from test_sealed_tools import _sealable

    from refactory_dsh import workspace as workspaces

    _sealable()
    run, pass_dir, _workspace = _layout(tmp_path)
    (pass_dir / ".refactory" / "pass.json").write_text(
        json.dumps({"task": f"{ifo.MODULE}:ToolEnvTask"}))
    other = run / "passes" / "data" / "b" / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(other)], check=True)
    monkeypatch.setenv(ifo.PYTHON_ENV, str(other / "bin" / "python"))
    seal.interpreter_dirs.cache_clear()
    with pytest.raises(workspaces.WorkspaceError, match="may not read"):
        workspaces.prepare(pass_dir, "look", "scratch", route={"worker_backend": "dsh"},
                           tools=("bash",))


def test_a_sealed_command_cannot_write_into_the_interpreter_s_folders(tmp_path):
    """[seal, Linux, sealed command] The interpreter's folders and the tool-bin are bound
    read only, though the person owns them."""
    from test_sealed_tools import _sealable

    _sealable()
    run, _pass, workspace = _layout(tmp_path)
    venv = tmp_path / "venvs" / "v"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    seal.interpreter_dirs.cache_clear()
    made = seal.tool_path(str(venv / "bin" / "python"), workspace.parent / "tool-bin",
                          workspace=workspace, runs=[run])
    assert str(venv) in made.reads
    result = _sealed(workspace, made, f"touch {venv}/planted {made.bin}/planted; "
                                      "python3 -c 'import sys; print(sys.prefix)'")
    assert not (venv / "planted").exists() and not (Path(made.bin) / "planted").exists()
    assert result.stdout.split() == [str(venv)], result


def test_dsh_s_own_sandbox_provider_runs_the_task_s_interpreter(tmp_path):
    """[TS: DSH's local sandbox provider, given the runner as its ``runnerCommand``] A
    sealed step prepared in an item's pass whose Task declares an environment: the bash
    command DSH wraps runs the Task's interpreter as python3, and still sees no grader
    file."""
    from test_sealed_tools import ROOT, _sealable

    from refactory_dsh import workspace

    _sealable()
    if not (ROOT / "node_modules" / ".pnpm").is_dir():
        pytest.skip("DSH is not installed here")
    run = tmp_path / "run"
    pass_dir = run / "passes" / "data" / "a"
    (pass_dir / ".refactory").mkdir(parents=True)
    (pass_dir / ".refactory" / "pass.json").write_text(
        json.dumps({"task": f"{ifo.MODULE}:ToolEnvTask"}))
    (run / "grader.json").write_text("GOLD-IN-RUN")
    (pass_dir / "input.txt").write_text("THE-INPUT")
    place = workspace.prepare(pass_dir, "probe", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("input.txt",))
    assert place is not None and place.tool_environment["summary"] == ifo.TOOL_SUMMARY
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "runner": list(place.runner), "workspace": str(place.path),
        "command": f"python3 -c 'import sys; print(sys.prefix)'; cat {run / 'grader.json'}",
        "failure": seal.FAILURE.strip()}))
    result = subprocess.run(["node", str(Path(__file__).with_name("sealed_tools_test.mjs")),
                             str(ROOT), str(request), "provider"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["enforcement"] == "full"
    assert observed["stdout"].strip() == _prefix(place.tool_environment["python"]), observed
    assert "GOLD" not in observed["stdout"] + observed["stderr"]
    workspace.collect(place, pass_dir, [])


def _venv(path: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True)
    seal.interpreter_dirs.cache_clear()
    return path / "bin" / "python"


def test_an_interpreter_in_a_task_s_hidden_path_is_refused_through_the_workspace(
        tmp_path, monkeypatch):
    """[both runtimes' `workspace.prepare`] The Task's `ToolEnvironment.hidden` reaches the
    seal: an interpreter installed inside the dataset it names is refused; outside, it is
    given."""
    from test_sealed_tools import _sealable

    from refactory_dsh import workspace as workspaces

    _sealable()
    _run, pass_dir, _workspace = _layout(tmp_path)
    (pass_dir / ".refactory" / "pass.json").write_text(
        json.dumps({"task": f"{ifo.MODULE}:HiddenEnvTask"}))
    dataset = tmp_path / "x" / "dataset"
    (dataset / "gold.txt").parent.mkdir(parents=True)
    (dataset / "gold.txt").write_text("GOLD-SECRET")
    monkeypatch.setenv(ifo.HIDDEN_ENV, str(dataset))
    monkeypatch.setenv(ifo.PYTHON_ENV, str(_venv(dataset / "venv")))
    with pytest.raises(workspaces.WorkspaceError, match="may not read"):
        workspaces.prepare(pass_dir, "look", "scratch", route={"worker_backend": "dsh"},
                           tools=("bash",))
    monkeypatch.setenv(ifo.PYTHON_ENV, str(_venv(tmp_path / "elsewhere" / "venv")))
    place = workspaces.prepare(pass_dir, "look", "scratch", route={"worker_backend": "dsh"},
                               tools=("bash",))
    assert place is not None and place.tool_environment["task"].endswith(":HiddenEnvTask")


def test_a_nested_pass_refuses_an_interpreter_in_a_sibling_item_s_pass(tmp_path, monkeypatch):
    """[both runtimes' `workspace.prepare`, `items.run_roots`] A pass nested in an outer
    item's pass is protected as the whole run: a venv in another outer item's pass is
    refused, with no run record above to find the run by."""
    from test_sealed_tools import _sealable

    from refactory_dsh import workspace as workspaces
    from refactory_dsh.items import PASS_MARKER, run_roots

    _sealable()
    run = tmp_path / "run"
    outer = run / "passes" / "data" / "a"
    inner = outer / "passes" / "inner" / "x"
    for folder in (outer, inner):
        (folder / ".refactory").mkdir(parents=True)
        (folder / PASS_MARKER).write_text(json.dumps({"task": f"{ifo.MODULE}:ToolEnvTask"}))
    assert run_roots(inner)[0] == str(run.resolve())
    monkeypatch.setenv(ifo.PYTHON_ENV, str(_venv(run / "passes" / "data" / "b" / "venv")))
    with pytest.raises(workspaces.WorkspaceError, match="may not read"):
        workspaces.prepare(inner, "look", "scratch", route={"worker_backend": "dsh"},
                           tools=("bash",))
