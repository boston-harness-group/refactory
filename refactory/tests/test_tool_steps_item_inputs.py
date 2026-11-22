"""A tool step the outer loop adds in the per-item part works on the item's input files,
and runs its commands with the Task's declared tool environment.

- **Item inputs by default** (`core_ops._with_item_files`): a sealed tool step a core
  operator adds or changes inside a per-item pass gets the item's input files
  (`Task.item_files`) in its declared ``reads``, beside what it names; the graph says so
  and so does the candidate's notes. Outside a pass, or without tools, nothing is added.
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
    assert any(f"look (tools, in the per-item part) is given the item's input file(s) "
               f"{ifo.INPUT}" in note for note in out.notes), out.notes
    out = _attempt(mode, [{"op": "fallback", "step": "solve", "kind": "agent",
                           "tools": ["bash"]}])
    assert ifo.INPUT in _nodes(out)["solve_fallback"]["reads"]


def test_a_tool_step_that_names_the_input_keeps_it_once():
    """[edit layer] What the step names stays; the input is not added twice, and nothing
    is said to be given that it named itself... once it is there already."""
    out = _attempt(ifo.AgentPass(), [{**LOOK, "sees": [ifo.INPUT]}])
    assert _nodes(out)["look"]["reads"] == [ifo.INPUT]


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
    assert any("explore (tools, in the per-item part) is given the item's input file(s) "
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


def test_a_rewired_tool_step_keeps_its_tools_and_its_item_inputs():
    """[edit layer] A rewire of a sealed step rebuilds it: it still runs sealed (its tool
    policy is not reset to a new step's 'none') and still reads the item's input."""
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
    assert ifo.INPUT in look["reads"], look


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
    assert "in the per-item part the item's input files too" in on
    assert "item's input files" not in off
    # A Task that declares no environment: no such line, tools or not.
    plain = _operation_text(ifo.ItemFilesMode().state_dict(), tools=True)
    assert "runs its commands with" not in plain


def test_the_analysts_get_the_same_fact():
    """[request text] The analysts' "changes the optimizer can make" is the same lines."""
    from types import SimpleNamespace

    from refactory_dsh.optim.analysts import _changes_text
    from refactory_dsh.optim.memory import state_view

    view = state_view(sb_mode.Solver().state_dict())
    ctx = SimpleNamespace(catalogue=nn_edits.catalogue("broad"), node_states=view.node_states,
                          edges=view.edges, knobs=[], new_node_tools=True)
    assert sb_interpreter.SUMMARY in _changes_text(ctx, "success")
    ctx.new_node_tools = False
    assert sb_interpreter.SUMMARY not in _changes_text(ctx, "success")


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


def test_the_seal_never_binds_an_interpreter_folder_that_holds_the_run_or_home(tmp_path):
    """[seal] An interpreter whose folder holds the run (or the home directory) is
    refused; its folder beside the workspace, never inside it."""
    run = tmp_path / "run"
    workspace = run / ".refactory" / "workspaces" / "look" / "scratch-1"
    workspace.mkdir(parents=True)
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    seal.interpreter_dirs.cache_clear()
    with pytest.raises(seal.SealError, match="holds"):
        seal.tool_path(str(python), workspace.parent / "tool-bin", workspace=workspace, run=run)
    with pytest.raises(seal.SealError, match="beside the workspace"):
        seal.tool_path(sys.executable, workspace / "tool-bin", workspace=workspace, run=run)
    made = seal.tool_path(sys.executable, workspace.parent / "tool-bin", workspace=workspace,
                          run=run)
    assert sorted(p.name for p in Path(made.bin).iterdir()) == ["python", "python3"]
    assert made.bin in made.reads


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
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "pass.json").write_text(json.dumps({"task": f"{ifo.MODULE}:ToolEnvTask"}))
    (run / "grader.json").write_text("GOLD-IN-RUN")
    (run / "input.txt").write_text("THE-INPUT")
    place = workspace.prepare(run, "probe", "scratch", route={"worker_backend": "dsh"},
                              tools=("bash",), reads=("input.txt",))
    assert place is not None and place.tool_environment["summary"] == ifo.TOOL_SUMMARY
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "runner": list(place.runner), "workspace": str(place.path),
        "command": "python3 -c 'import sys; print(sys.prefix)'; cat ../../../../grader.json",
        "failure": seal.FAILURE.strip()}))
    result = subprocess.run(["node", str(Path(__file__).with_name("sealed_tools_test.mjs")),
                             str(ROOT), str(request), "provider"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["enforcement"] == "full"
    assert observed["stdout"].strip() == _prefix(place.tool_environment["python"]), observed
    assert "GOLD" not in observed["stdout"] + observed["stderr"]
    workspace.collect(place, run, [])
