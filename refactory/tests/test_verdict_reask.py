"""A judge whose answer names no verdict is asked again for the word, on both runtimes.

A verdict gate (`snippet.verdict`) reads its verdict from an agent's answer. A judge that
reviews first and never puts the word first, or alone last ("Let me inspect ...", "I found
critical row-mapping errors ..."), used to fail its item: the SpreadsheetBench V2
shakedown (2026-11-23) lost 6 of 66 train items so, on exactly the items where the judge
found a problem. Now the agent is asked once more, in a fresh call with its own answer,
for the word alone (`node_retry.settle_verdict`); if that names none either, the gate's
declared way on is taken -- its ``proceed`` when its loop goes on there when it cannot
settle (``on_exhausted: continue``) -- and the file says so. Both runtimes settle the
answer in `node_retry.deliver`: the walker from `Executor._visit_agent`, the spine from
the worker each agent node runs (`worker_node.execute`), whose gate then runs through
the node runner (`python -m refactory_dsh.run`).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh import lowering, snippet, workers
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, Node
from refactory_dsh.nn.loops import LOOP_END_CALL
from refactory_dsh.node_retry import VERDICT_REASK_EVENT

RUNTIMES = ["walker", "spine"]
JUDGE = "Judge the work."
#: Text only the re-ask's prompt holds (`snippet.VERDICT_REASK`).
REASK = "Reply with only the one word your answer decides"
PROSE = ("Let me inspect the sheet first.\nI found critical row-mapping errors: row 4 maps "
         "to column C, not B.")

FAKE_DSH = '''import json, pathlib, sys
plans = json.loads(pathlib.Path({plan!r}).read_text())
prompt = sys.stdin.read()
key = next(key for key in plans if key in prompt)
log = pathlib.Path({log!r})
calls = json.loads(log.read_text()) if log.exists() else []
plan = plans[key]
answer = plan[min(sum(1 for c in calls if c["key"] == key), len(plan) - 1)]
calls.append({{"key": key, "prompt": prompt}})
log.write_text(json.dumps(calls))
print(answer)
'''


class Scripted:
    """A scripted `dsh` answering each prompt by the first key it contains, in turn."""

    def __init__(self, root: Path, plans: dict[str, list[str]]):
        root.mkdir(parents=True, exist_ok=True)
        self.log = root / "calls.json"
        (root / "plan.json").write_text(json.dumps(plans))
        script = root / "fake_dsh.py"
        script.write_text(FAKE_DSH.format(plan=str(root / "plan.json"), log=str(self.log)))
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)

    def calls(self, key: str | None = None) -> list[dict]:
        made = json.loads(self.log.read_text()) if self.log.exists() else []
        return [c for c in made if key is None or c["key"] == key]

    def client(self) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=60, inactivity_s=30)


def _graph(on_exhausted: str = "continue") -> Graph:
    """``judge -> check``; ``check`` goes on to ``done`` on proceed, back to ``judge`` on
    reloop (once)."""
    graph = Graph()
    graph.add(Node("judge", "agent", prompt_template=JUDGE, writes=("judge.txt",),
                   tool_policy="none"))
    graph.add(Node("check", "gate", evaluator=snippet.VERDICT, reads=("judge.txt",),
                   required_reads=("judge.txt",), max_iterations=1, on_exhausted=on_exhausted))
    graph.add(Node("done", "op", call=LOOP_END_CALL, score_effect="none"))
    graph.connect("judge", "check")
    graph.connect("check", "done", condition="proceed")
    graph.connect("check", "judge", condition="reloop")
    return graph


def _events(workdir: Path) -> list[dict]:
    path = workdir / ".refactory/events.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [e for e in map(json.loads, lines) if e.get("kind") == VERDICT_REASK_EVENT]


def _walk(workdir: Path, graph: Graph, dsh: Scripted):
    from refactory_dsh.nn.executor import Executor, RunContext

    workdir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(workdir, knobs={}, facts={}, task={})
    ctx.model = dsh.client()
    walker = Executor(graph, ctx)
    result = walker.run(start=graph.start)
    result.verdicts = [s.verdict for s in walker.steps if s.node == "check"]
    return result


def _spine(workdir: Path, graph: Graph, dsh: Scripted, monkeypatch, capsys
           ) -> tuple[int, str, str]:
    """The judge's worker command as the spine lowers it, then the gate through the node
    runner: (the gate's exit code, its stdout, its stderr)."""
    from refactory_dsh import run as run_mod

    workdir.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state(), "knobs": {}}
    lowering.write_declarations(state, workdir)
    lowered = workers.lower_workflow(lowering.to_workflow(state), workdir, dsh.client().context(),
                                     lowering.node_declarations(state))
    command = lowered.nodes["judge"].dispatch_command
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    done = subprocess.run(shlex.split(command), cwd=workdir, env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "check", "type": "gate"}))
    monkeypatch.setenv("REFACTORY_WORKDIR", str(workdir))
    monkeypatch.setenv("REFACTORY_PROJECT", str(workdir))
    capsys.readouterr()
    code = run_mod.main([snippet.VERDICT])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_judge_that_answers_in_prose_is_asked_again_and_its_reloop_is_honoured(
        tmp_path, runtime, monkeypatch, capsys):
    dsh = Scripted(tmp_path / "dsh", {REASK: ["reloop"], JUDGE: [PROSE, "proceed: fixed"]})
    workdir = tmp_path / "run"
    if runtime == "walker":
        result = _walk(workdir, _graph(), dsh)
        assert result.exit_ok, result.error
        # The reloop happened: the judge ran twice; its second answer named a verdict.
        assert len(dsh.calls(JUDGE)) == 2
        assert result.verdicts == ["reloop", "proceed"]
    else:
        code, out, err = _spine(workdir, _graph(), dsh, monkeypatch, capsys)
        assert code == 0, err
        lines = out.strip().splitlines()
        assert lines[-1] == "reloop"
        # The reason a reloop hands back says how the verdict was read, then the review.
        reason = "\n".join(lines[:-1])
        assert "asked again for the word alone, judge answered 'reloop'" in reason
        assert "row 4 maps to column C" in reason
        text = (workdir / "judge.txt").read_text()
        assert text.startswith("reloop\n[refactory: ") and text.endswith(PROSE + "\n")
    [reask] = dsh.calls(REASK)
    assert PROSE in reask["prompt"] and "proceed or reloop" in reask["prompt"]
    [event] = _events(workdir)
    assert (event["node_id"], event["state"], event["verdict"], event["gate"]) == (
        "judge", "reasked", "reloop", "check")


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_still_unreadable_after_the_reask_takes_the_gates_way_on_and_says_so(
        tmp_path, runtime, monkeypatch, capsys):
    dsh = Scripted(tmp_path / "dsh", {REASK: ["Hard to say."], JUDGE: [PROSE]})
    workdir = tmp_path / "run"
    if runtime == "walker":
        result = _walk(workdir, _graph(), dsh)
        assert result.exit_ok, result.error
        assert len(dsh.calls(JUDGE)) == 1
        assert result.verdicts == ["proceed"]
    else:
        code, out, err = _spine(workdir, _graph(), dsh, monkeypatch, capsys)
        assert code == 0, err
        assert out.strip().splitlines()[-1] == "proceed"
    text = (workdir / "judge.txt").read_text()
    assert text.startswith("proceed\n[refactory: the answer below named no verdict, also when "
                           "judge was asked again (it answered 'Hard to say.'); gate 'check' "
                           "takes 'proceed'"), text
    [event] = _events(workdir)
    assert (event["state"], event["verdict"]) == ("unreadable", "proceed")
    # The delivery receipt holds the settled file: the step's delivery is current.
    from refactory_dsh.agent_delivery import receipts_in

    receipt = receipts_in(workdir)["judge"]
    assert receipt["current"] and receipt["verdict_settled"] == "unreadable"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_gate_that_declares_halt_still_fails_on_an_unreadable_verdict(
        tmp_path, runtime, monkeypatch, capsys):
    """No way on is invented: a gate whose loop halts when it cannot settle fails as it did,
    after the one re-ask, and the event says why."""
    dsh = Scripted(tmp_path / "dsh", {REASK: ["Hard to say."], JUDGE: [PROSE]})
    workdir = tmp_path / "run"
    graph = _graph(on_exhausted="halt")
    if runtime == "walker":
        result = _walk(workdir, graph, dsh)
        assert not result.exit_ok
        assert "names no verdict" in str(result.error) or "is one of its answers" in str(result.error)
    else:
        code, _out, err = _spine(workdir, graph, dsh, monkeypatch, capsys)
        assert code == 1 and "is one of its answers" in err, err
    assert (workdir / "judge.txt").read_text().strip() == PROSE
    [event] = _events(workdir)
    assert event["state"] == "failed"
    assert len(dsh.calls(REASK)) == 1


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_judge_that_names_its_verdict_is_not_asked_again(tmp_path, runtime, monkeypatch, capsys):
    dsh = Scripted(tmp_path / "dsh", {REASK: ["reloop"],
                                      JUDGE: [PROSE + "\n\nVERDICT: proceed"]})
    workdir = tmp_path / "run"
    if runtime == "walker":
        result = _walk(workdir, _graph(), dsh)
        assert result.exit_ok, result.error
    else:
        code, out, err = _spine(workdir, _graph(), dsh, monkeypatch, capsys)
        assert code == 0 and out.strip().splitlines()[-1] == "proceed", err
    assert dsh.calls(REASK) == [] and _events(workdir) == []
    assert (workdir / "judge.txt").read_text().strip().endswith("VERDICT: proceed")


def test_only_the_agent_a_verdict_gate_reads_is_settled():
    """`lowering.judge_verdict`: the agent whose first write is the gate's first required
    read, with the gate's labels and its way on; nothing for another node, or for an
    agent two verdict gates read (which answers to ask for is not one question)."""
    graph = _graph()
    assert lowering.judge_verdict(graph, "judge") == {
        "gate": "check", "labels": ["proceed", "reloop"], "unreadable": "proceed"}
    assert lowering.judge_verdict(graph, "check") is None
    assert lowering.judge_verdict(graph, "done") is None
    # Outside a data pass an undeclared gate halts when it cannot settle: no way on.
    undeclared = _graph(on_exhausted="")
    assert lowering.judge_verdict(undeclared, "judge")["unreadable"] == ""
    graph.add(Node("check2", "gate", evaluator=snippet.VERDICT, required_reads=("judge.txt",),
                   reads=("judge.txt",)))
    assert lowering.judge_verdict(graph, "judge") is None
    # The spine's worker reads it from the declarations the lowering writes.
    state = {"graph": _graph().to_state(), "knobs": {}}
    assert lowering.node_declarations(state)["judge"]["verdict"]["gate"] == "check"
    assert "verdict" not in lowering.node_declarations(state)["done"]


def test_a_loop_judge_in_a_data_pass_proceeds_when_its_verdict_stays_unreadable():
    """The ``loop`` operator's judge, inside a data pass, is settled with the gate's
    ``proceed`` as its way on (the loop declares ``on_exhausted: continue``); its prompt asks
    for the word alone on the first line, with an example."""
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.nn.core_ops import system_appended

    state = sf.Applier(sf.seeds()["fuzz_draft_review"]).seed_state
    state = nn_edits.apply({"op": "loop", "step": "draft", "judge": "agent",
                            "text": "Check the draft."}, state)
    graph = Graph.from_state(state["graph"])
    judge = next(n for n in graph.nodes if n.startswith("draft_check_judge"))
    found = lowering.judge_verdict(graph, judge)
    assert found == {"gate": "draft_check", "labels": ["proceed", "reloop"],
                     "unreadable": "proceed"}
    knob = dict(graph.nodes[judge].field_bindings)["prompt_template"]
    own, rule = system_appended(state["knobs"][knob]["value"])
    assert own.startswith("Check the draft.")
    assert "first line is one word alone: proceed or reloop" in rule
    assert 'For example, a first line "reloop"' in rule and "\n" not in rule


def test_an_earlier_verdict_rule_is_still_said_apart():
    """A state built before the rule changed keeps its judge's old rule in the prompt; the
    view still shows it as appended by the system."""
    from refactory_dsh.nn.core_ops import system_appended

    old = ("Check.\n\nStart your answer with one word, proceed or reloop, alone or followed "
           "by why; nothing before it.")
    assert system_appended(old) == ("Check.", old.split("\n\n", 1)[1])
    assert system_appended("Check.\n\nproceed or reloop?") == ("Check.\n\nproceed or reloop?", "")


@pytest.mark.parametrize("text, expected", [
    ("Let me inspect first.", False),
    ("reloop: row 4", True),
    ("Found the issue.\n\nVERDICT: reloop", True),
    ("Proceed with care.\n\nreloop", None),
])
def test_names_verdict_reads_as_the_gate_reads(text, expected):
    assert snippet.names_verdict(text, ["proceed", "reloop", "halt"]) is expected


def test_the_reask_quotes_a_long_answer_by_its_start_and_end():
    text = "A" * 5000 + "MIDDLE" + "Z" * 9000
    prompt = snippet.reask_prompt(text, ["proceed", "reloop"])
    assert "MIDDLE" not in prompt and "characters left out" in prompt
    assert prompt.count("A") >= snippet.REASK_HEAD_CHARS
    assert "Z" * snippet.REASK_TAIL_CHARS in prompt
