"""A ``loop`` or ``route`` judge may have tools, as a step ``insert`` adds may.

The SpreadsheetBench V2 shakedown (2026-11-23): the loop's judge had no tools
(``tool_policy: none``), its files pasted in as text, so it could not open the output
workbook or run the solution; its answers began "Let me inspect the input spreadsheet...",
and it passed 33 of 51 failures. Now an agent judge takes 'tools' through the same path as
an inserted step (`core_ops._step_nodes`): sealed, in a fresh workspace with the files it
sees -- without 'sees', in the per-item part, the item's input files too -- under the
attempt's tool rules (``E-new-node-tools``) and the Task's tool environment; its reply is
still the verdict its gate reads.
"""
from __future__ import annotations

import sys

import pytest

import item_files_ops as ifo
from refactory_dsh import lowering, workers
from refactory_dsh.nn import core_ops
from refactory_dsh.nn.graph import Graph
from refactory_dsh.optim import Candidate, Optimizer, Proposal
from refactory_dsh.workspace import is_sealed
from test_livemath_blocks import _spine, needs_spine

RUNTIMES = ["walker", pytest.param("spine", marks=needs_spine)]
LOOP = {"op": "loop", "step": "solve", "judge": "agent", "text": "Check the answer.",
        "tools": ["bash"]}
ROUTE = {"op": "route", "after": "solve", "judge": "agent", "text": "Choose.",
         "tools": ["bash"]}
#: A step after solve in the pass, so a route after solve has a next step to go on to.
POLISH = {"op": "insert", "after": "solve", "id": "polish", "kind": "agent",
          "text": "Polish the answer.", "takes_over": "solve"}


def _attempt(edits: list[dict], *, sealed: bool = True, mode=None):
    mode = mode or ifo.AgentPass()
    optimizer = Optimizer()
    optimizer.sealed_tools = (sealed, "" if sealed else "no sandbox here")
    seed = Candidate.from_state(mode.state_dict())
    return optimizer.attempt(Proposal(edits=edits), seed, mode.invariants(), True, (),
                             protected=mode.edit_protected(), record=False)


def _nodes(out) -> dict:
    assert isinstance(out, Candidate), out
    return out.state["graph"]["nodes"]


def test_a_loop_judge_with_tools_is_a_sealed_step_with_the_item_input_and_the_step_s_output():
    out = _attempt([LOOP])
    judge = _nodes(out)["solve_check_judge"]
    assert judge["tools"] == ["bash"] and is_sealed(judge)
    assert judge["workspace"]["kind"] == "scratch" and judge.get("tool_policy") is None
    # What the step wrote and the item's input file, in its sealed workspace: no collect
    # step pastes them in as text.
    assert set(judge["reads"]) == {ifo.ANSWER, ifo.INPUT}, judge["reads"]
    assert "solve_check_judge_inputs" not in _nodes(out)
    # Its reply is the verdict its gate reads, and it is told so.
    gate = _nodes(out)["solve_check"]
    assert gate["required_reads"] == ["solve_check_judge.txt"]
    assert judge["writes"] == ["solve_check_judge.txt"]
    text = out.state["knobs"][dict(judge["field_bindings"])["prompt_template"]]["value"]
    assert "first line is one word alone: proceed or reloop" in text


def test_without_tools_a_loop_judge_is_as_before():
    nodes = _nodes(_attempt([{k: v for k, v in LOOP.items() if k != "tools"}]))
    judge = nodes["solve_check_judge"]
    assert judge["tool_policy"] == "none" and not judge.get("tools")
    assert not is_sealed(judge)
    assert nodes["solve_check_judge_inputs"]["reads"] == [ifo.ANSWER]


def test_a_judge_whose_sees_names_its_files_gets_exactly_those():
    judge = _nodes(_attempt([{**LOOP, "sees": [ifo.ANSWER]}]))["solve_check_judge"]
    assert judge["reads"] == [ifo.ANSWER]


def test_a_route_judge_takes_tools_as_well():
    nodes = _nodes(_attempt([POLISH, ROUTE]))
    judge = next(n for i, n in nodes.items() if i.endswith("_judge"))
    assert judge["tools"] == ["bash"] and is_sealed(judge)
    # The item's input, and what solve wrote (renamed by polish's take-over).
    assert set(judge["reads"]) == {ifo.INPUT, *nodes["solve"]["writes"]}, judge["reads"]
    plain = _nodes(_attempt([POLISH, {k: v for k, v in ROUTE.items() if k != "tools"}]))
    judge = next(n for i, n in plain.items() if i.endswith("_judge"))
    assert judge["tool_policy"] == "none" and not judge.get("tools")


def test_judge_tools_are_checked_as_insert_s_are():
    """The same list check, the same refusal where the machine cannot seal a step, and no
    tools for a code judge."""
    refused = _attempt([{**LOOP, "tools": ["python"]}])
    assert isinstance(refused, list) and "'tools' lists the tools a new step uses" in refused[0]
    refused = _attempt([{**LOOP, "judge": "code", "text": None}])
    assert isinstance(refused, list) and "a 'code' judge runs no tools" in refused[0], refused
    refused = _attempt([LOOP], sealed=False)
    assert isinstance(refused, list) and refused[0].startswith("E-new-node-tools"), refused


def test_the_catalogue_offers_judge_tools_only_where_tools_are_offered():
    for name in ("loop", "route"):
        entry = core_ops.OPERATORS[name]
        assert "tools" in entry["fields"] and "'tools'" in entry["line"]
        line, fields = core_ops.without_tools(entry["line"], entry["fields"])
        assert "tools" not in fields and "'tools'" not in line


# ── on both runtimes: the sealed judge reads the item and the answer, then decides ──


#: A stand-in DSH CLI. The judge (sealed: its patch names the sandbox) reads the item's
#: input and the step's answer with bash, in its workspace, and decides; any other step
#: answers with the item's input it was handed nothing of.
JUDGE_AGENT = """\
import json, pathlib, subprocess, sys
patch = []
if '--patch' in sys.argv:
    last = max(i for i, arg in enumerate(sys.argv) if arg == '--patch')
    patch = json.loads(pathlib.Path(sys.argv[last + 1]).read_text())
runner = next((row['config']['runnerCommand'] for row in patch if row.get('id') == 'sandbox'),
              None)
if runner is None:
    print('my answer')
    raise SystemExit(0)
def bash(script):
    done = subprocess.run([*runner, '--ro-bind', '/', '/', '--', 'bash', '-c', script],
                          cwd=pathlib.Path.cwd(), capture_output=True, text=True, timeout=120)
    return done.stdout.strip() or 'DENIED: ' + done.stderr.strip()[-300:]
print('proceed')
print('input=' + bash('cat item/input.bin'))
print('answer=' + bash('cat answer.txt'))
"""


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_sealed_loop_judge_reads_the_item_and_the_answer_and_its_verdict_routes(
        tmp_path, runtime):
    """[walker; spine, the judge dispatched through `worker_node`]"""
    from test_sealed_tools import _sealable

    from refactory_dsh.model import DshModelClient

    _sealable()
    mode = ifo.AgentPass()
    out = _attempt([LOOP], mode=mode)
    assert isinstance(out, Candidate), out
    mode.load_state_dict(out.state)
    script = tmp_path / "agent-cli"
    script.write_text(f"#!{sys.executable}\n" + JUDGE_AGENT)
    script.chmod(0o755)
    client = DshModelClient(worker_backend="dsh", dsh_cli=str(script), attempts=1)
    if runtime == "walker":
        result = mode.run(tmp_path / "walker", model=client)
        assert result.exit_ok, result.error
    else:
        state = mode.state_dict()
        rc = _spine(Graph.from_state(state["graph"]), tmp_path / "spine", tmp_path,
                    mode_state=state, agent_responses={},
                    transform=lambda wf: workers.lower_workflow(
                        wf, tmp_path / "specs", client.context(),
                        lowering.node_declarations(state)))
        assert rc == 0
    judged = sorted(tmp_path.glob(f"{runtime}/passes/*/*/solve_check_judge.txt"))
    assert len(judged) == 2, judged
    for path in judged:
        text = path.read_text()
        assert text.startswith("proceed\n"), text
        assert "input=INPUT-" in text and "answer=my answer" in text, text
        # Proceeded once: the judge was not sent back, nor asked again.
        assert "[refactory:" not in text


def test_a_default_view_never_hands_on_a_sealed_session_log():
    """SpreadsheetBench V2's solve writes `.refactory-session.json` (its tool-call log,
    which nothing reads): a judge or step seeing solve by default does not get it."""
    from refactory_packages.spreadsheetbench_v2.swe_mode import Swe

    from refactory_dsh.workspace import SEALED_SESSION

    for edit in ({"op": "loop", "step": "solve", "judge": "agent"},
                 {"op": "loop", "step": "solve", "judge": "agent", "tools": ["bash"]}):
        nodes = _nodes(_attempt([edit], mode=Swe()))
        seed = Swe().state_dict()["graph"]["nodes"]
        reads = [r for k, n in nodes.items() if k not in seed for r in (n.get("reads") or ())]
        assert "output.xlsx" in reads and SEALED_SESSION not in reads, (edit, reads)
