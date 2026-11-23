"""Declared node recovery: a timed-out agent session is run once more, fresh and longer.

The rule is the graph's (`Node.retry`), and both runtimes run it through
`node_retry.deliver`: the walker from `Executor._visit_agent`, the spine from the worker
each dispatched agent node runs (`python -m refactory_dsh.worker_node <spec>`), inside the
envelope `workers.lower_workflow` reserves. The runtime tests here drive the real model
client against a scripted `dsh` that answers call by call, so a timeout is a real
deadline, not a raised string.

A timeout is a session gone silent past the inactivity limit (1 s here, for production's
600 s): a scripted step that sleeps streams nothing, and one that is ``active`` streams a
frame every 0.2 s, as the worker observer records it (`model.activity_path`). A session
that reaches its node's declared wall is the pipeline's own limit (`model.DeclaredTimeout`)
and is not retried.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from refactory_dsh import lowering, workers
from refactory_dsh.model import NODE_DISPATCH_OVERHEAD_S, CompletionTimeout, DshModelClient
from refactory_dsh.nn import Graph, GraphError, Node
from refactory_dsh.nn.graph import retry_declaration
from refactory_dsh.node_retry import (INVENTED_TOOL_CALLS, RETRY_EVENT, RetriesExhausted,
                                      attempt_walls, deliver, failure_kind, node_allowance_s,
                                      retries)

RULE = {"on": ["timeout"], "times": 1, "timeout_factor": 2}
REPORT = "reviews/report.md"

FAKE_DSH = '''import json, os, sys, time, pathlib
plans = json.loads(pathlib.Path({plan!r}).read_text())
prompt = sys.stdin.read()
import signal
key = next((key for key in plans if key and key in prompt), "")
counter = pathlib.Path({counter!r} + ("." + "".join(c for c in key if c.isalnum()) if key else ""))
calls = json.loads(counter.read_text()) if counter.exists() else 0
plan = plans[key]
step = plan[min(calls, len(plan) - 1)]
counter.write_text(json.dumps(calls + 1))
if step.get("fresh"):
    assert not pathlib.Path("stale.txt").exists(), "previous attempt survived"
    assert not pathlib.Path("reviews/report.md").exists(), "partial report survived"
    assert "Your workspace is " + str(pathlib.Path.cwd()) in prompt, "missing workspace contract"
for name, text in step.get("write", {{}}).items():
    target = pathlib.Path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
if step.get("ignore_term"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
activity = pathlib.Path(os.environ["REFACTORY_WORKER_RECEIPT"]).with_suffix(".activity")
end = time.time() + step.get("active", 0)
while time.time() < end:
    activity.touch()
    time.sleep(0.2)
time.sleep(step.get("sleep", 0))
if "exit" in step:
    print(step.get("stderr", ""), file=sys.stderr)
    sys.exit(step["exit"])
print(step.get("answer", "the report"))
'''


class Scripted:
    """A scripted `dsh`: what each call does, in order (the last step repeats). A plan
    keyed by prompt text answers each kind of prompt from its own list, so concurrent
    branches cannot take each other's steps."""

    def __init__(self, root: Path, plan: list[dict] | dict[str, list[dict]]):
        root.mkdir(parents=True, exist_ok=True)
        self.plans = plan if isinstance(plan, dict) else {"": plan}
        self.plan, self.counter = root / "plan.json", root / "calls.json"
        self.plan.write_text(json.dumps(self.plans))
        script = root / "fake_dsh.py"
        script.write_text(FAKE_DSH.format(plan=str(self.plan), counter=str(self.counter)))
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)

    @property
    def calls(self) -> int:
        return json.loads(self.counter.read_text()) if self.counter.exists() else 0

    def client(self, **extra) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=60, **{"inactivity_s": 1, **extra})


def _node(**fields) -> Node:
    defaults = dict(prompt_template="Report.", writes=(REPORT,), timeout_s=2, retry=RULE,
                    outputs_only=True)
    defaults.update(fields)
    return Node("reviewer", "agent", **defaults)


def _graph(*nodes: Node) -> Graph:
    graph = Graph()
    for node in nodes:
        graph.add(node)
    for before, after in zip(nodes, nodes[1:]):
        graph.connect(before.id, after.id)
    return graph


def _events(workdir: Path) -> list[dict]:
    path = workdir / ".refactory/events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _retried(workdir: Path) -> list[dict]:
    return [event for event in _events(workdir) if event.get("kind") == RETRY_EVENT]


# ── the two runtimes, each through its real entry point ─────────────────────────────


def _walk(workdir: Path, graph: Graph, client, budget=None):
    from refactory_dsh.nn.executor import Executor, RunContext

    workdir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(workdir, knobs={}, facts={}, task={})
    ctx.model = client
    ctx.budget = budget
    return Executor(graph, ctx).run(start=graph.start)


def _dispatch(workdir: Path, graph: Graph, context: dict) -> tuple[subprocess.CompletedProcess, int]:
    """Lower ``graph`` as the spine does and run its first agent's worker command."""
    workdir.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state(), "knobs": {}}
    lowering.write_declarations(state, workdir)
    lowered = workers.lower_workflow(lowering.to_workflow(state), workdir, context,
                                     lowering.node_declarations(state))
    node = lowered.nodes[graph.start]
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    done = subprocess.run(shlex.split(node.dispatch_command), cwd=workdir, env=env,
                          capture_output=True, text=True, timeout=node.command_timeout_s)
    return done, node.command_timeout_s


def _run(runtime: str, workdir: Path, graph: Graph, client: DshModelClient, budget=None):
    """Run ``graph``'s first node on ``runtime``; return (succeeded, error text)."""
    if runtime == "walker":
        result = _walk(workdir, graph, client, budget)
        return result.exit_ok, result.error or ""
    context = client.context()
    if budget is not None:
        context["budget_ledger"] = str(budget.path)
    done, _ = _dispatch(workdir, graph, context)
    return done.returncode == 0, done.stderr


RUNTIMES = ["walker", "spine"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_session_that_times_out_once_is_retried_with_the_longer_limit(tmp_path, runtime):
    # The first session writes part of its report and goes silent; the retry streams for
    # 3 s, which only the doubled 4 s limit allows, and answers.
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30, "write": {REPORT: "half a report"}},
                                      {"active": 3, "answer": "the whole report"}])
    workdir = tmp_path / "run"
    ok, error = _run(runtime, workdir, _graph(_node()), dsh.client())
    assert ok, error
    assert dsh.calls == 2
    assert (workdir / REPORT).read_text().strip() == "the whole report"
    [event] = _retried(workdir)
    assert (event["node_id"], event["attempt"], event["attempts"], event["wall_s"]) == ("reviewer", 2, 2, 4)
    # The first attempt's partial report was set aside, not left for the retry to keep.
    receipts = [json.loads(p.read_text()) for p in (workdir / ".refactory/agent-delivery").glob("*/receipt.json")]
    assert sorted(r["status"] for r in receipts) == ["delivered", "superseded"]
    [kept] = list((workdir / ".refactory/agent-delivery").glob(f"*/previous/*/{REPORT}"))
    assert kept.read_text() == "half a report"
    assert list((workdir / ".refactory/agent-delivery/pending").glob("*.json")) == []
    assert retries(workdir) == {"reviewer": 1}


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_node_that_times_out_twice_stops_with_the_reason(tmp_path, runtime):
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30}])
    ok, error = _run(runtime, tmp_path / "run", _graph(_node(timeout_s=3)), dsh.client())
    assert not ok
    # The time each silent session actually ran: the inactivity limit, and a moment to
    # shut down.
    assert re.search(r"reviewer timed out twice \(after [12]s, then [12]s\)", error), error
    assert dsh.calls == 2


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_without_a_rule_a_timeout_is_not_retried(tmp_path, runtime):
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30}, {"answer": "late"}])
    ok, error = _run(runtime, tmp_path / "run", _graph(_node(timeout_s=3, retry=None)), dsh.client())
    assert not ok and "model completion timed out after 1s of inactivity" in error
    assert dsh.calls == 1
    assert _retried(tmp_path / "run") == []


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_failure_that_is_not_a_timeout_is_not_retried(tmp_path, runtime):
    dsh = Scripted(tmp_path / "dsh", [{"exit": 1, "stderr": "dsh: the key was refused"},
                                      {"answer": "second"}])
    ok, error = _run(runtime, tmp_path / "run", _graph(_node()), dsh.client())
    assert not ok and "the key was refused" in error
    assert dsh.calls == 1
    assert _retried(tmp_path / "run") == []


def _review_fork(verifier: Node) -> Graph:
    graph = Graph()
    graph.add(Node("fork", "fork", targets=("verifier", "reviewer"), optional_targets=("verifier",)))
    graph.add(verifier)
    graph.add(Node("reviewer", "agent", prompt_template="Review.", writes=("reviews/review.md",)))
    graph.add(Node("join", "join", sources=("reviewer",)))
    graph.connect("fork", "join")
    graph.connect("verifier", "join")
    graph.connect("reviewer", "join")
    return graph


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_optional_node_that_times_out_twice_is_missing_and_says_why(tmp_path, runtime):
    verifier = Node("verifier", "agent", prompt_template="Verify.", writes=("reviews/verification.md",),
                    timeout_s=3, retry=RULE, outputs_only=True)
    workdir = tmp_path / "run"
    if runtime == "walker":
        from refactory_dsh.nn.executor import Executor, RunContext

        # Both branches reach one scripted dsh, which answers each by its prompt.
        dsh = Scripted(tmp_path / "dsh", {"Verify.": [{"sleep": 30}], "": [{"answer": "fine"}]})
        workdir.mkdir()
        ctx = RunContext(workdir, knobs={}, facts={}, task={})
        ctx.model = dsh.client()
        result = Executor(_review_fork(verifier), ctx).run(start="fork")
        assert result.exit_ok, result.error
        calls = json.loads(next(p for p in (tmp_path / "dsh").glob("calls.json.*")).read_text())
    else:
        dsh = Scripted(tmp_path / "dsh", [{"sleep": 30}])
        graph = _review_fork(verifier)
        graph.start = "verifier"
        done, _ = _dispatch(workdir, graph, dsh.client().context())
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout.strip().splitlines()[-1])["status"] == "missing"
        calls = dsh.calls
    assert calls == 2
    assert "verifier timed out twice" in (workdir / "reviews/verifier-missing.txt").read_text()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_declared_wall_is_the_pipelines_limit_and_its_declared_retry_answers_it(tmp_path, runtime):
    """A session still streaming at its node's declared 2 s wall stops there as a
    `DeclaredTimeout`, recorded ``declared_timeout`` from either runtime's text
    (`items.error_kind`): the pipeline's own limit. The retry on timeout the node declares
    is the graph's choice too, so it answers the wall: with every attempt still streaming
    at its wall, the node is tried once more and then fails as ``declared_timeout``."""
    from refactory_dsh.items import error_kind

    dsh = Scripted(tmp_path / "dsh", [{"active": 30}, {"active": 30}, {"answer": "late"}])
    ok, error = _run(runtime, tmp_path / "run", _graph(_node(timeout_s=2)), dsh.client())
    assert not ok
    assert "declared timeout_s" in error
    assert error_kind(RuntimeError(error)) == "declared_timeout"
    assert dsh.calls == 1 + int(RULE["times"]) and len(_retried(tmp_path / "run")) == int(RULE["times"])


def _missing_verifier(runtime: str, tmp_path: Path, verifier: Node, plan: list[dict],
                      **client) -> dict:
    """Run ``verifier`` as the optional branch of `_review_fork` on ``runtime``; return its
    ``items.missing_steps`` entry."""
    from refactory_dsh.items import missing_steps

    workdir = tmp_path / "run"
    if runtime == "walker":
        from refactory_dsh.nn.executor import Executor, RunContext

        dsh = Scripted(tmp_path / "dsh", {"Verify.": plan, "": [{"answer": "fine"}]})
        workdir.mkdir()
        ctx = RunContext(workdir, knobs={}, facts={}, task={})
        ctx.model = dsh.client(**client)
        result = Executor(_review_fork(verifier), ctx).run(start="fork")
        assert result.exit_ok, result.error
    else:
        dsh = Scripted(tmp_path / "dsh", plan)
        graph = _review_fork(verifier)
        graph.start = "verifier"
        done, _ = _dispatch(workdir, graph, dsh.client(**client).context())
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout.strip().splitlines()[-1])["status"] == "missing"
    return missing_steps(workdir)["verifier"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_optional_branch_says_how_it_ended_for_time(tmp_path, runtime):
    """An optional branch missing because it ended for time says how, so a reader (and
    #684's harness-timeout reading) tells the harness's ending from the pipeline's: its
    declared wall is ``declared_timeout``, the harness backstop ``backstop``,
    and silence ``timeout``. Any other failure stays ``failed``."""
    def verifier(**fields) -> Node:
        return Node("verifier", "agent", prompt_template="Verify.",
                    writes=("reviews/verification.md",), outputs_only=True, **fields)

    walled = _missing_verifier(runtime, tmp_path / "wall", verifier(timeout_s=2), [{"active": 30}])
    assert walled["cause"] == "declared_timeout", walled
    stopped = _missing_verifier(runtime, tmp_path / "backstop", verifier(), [{"active": 30}],
                                backstop_s=3)
    assert stopped["cause"] == "backstop", stopped
    assert "hit the harness backstop after 3s" in stopped["error"]
    silent = _missing_verifier(runtime, tmp_path / "silent", verifier(), [{"sleep": 30}])
    assert silent["cause"] == "timeout", silent
    failed = _missing_verifier(runtime, tmp_path / "failed", verifier(),
                               [{"exit": 1, "stderr": "dsh: the key was refused"}])
    assert failed["cause"] == "failed", failed


def test_the_qa_judge_and_overwatch_read_that_the_verifier_is_missing(tmp_path):
    """The QA synthesis is what both judges read: it says the verifier is missing and why."""
    from refactory_dsh.design_ops import QA_SYNTHESIS, synthesize_qa
    from refactory_dsh.nn.executor import RunContext

    (tmp_path / "reviews").mkdir()
    (tmp_path / "reviews/verifier-missing.txt").write_text(
        "RetriesExhausted: verifier timed out twice (after 1200s, then 2400s); its declared "
        "retry is spent.\n")
    ctx = RunContext(tmp_path, knobs={}, facts={"check": "pytest"}, task={})
    synthesize_qa(ctx)
    body = (tmp_path / QA_SYNTHESIS).read_text()
    assert "the design-blind verifier is missing (timed out twice)" in body
    assert "VERIFIER MISSING (timed out twice)" in body
    assert "after 1200s, then 2400s" in body
    # A verifier that reported is read as before, and nothing says it is missing.
    (tmp_path / "reviews/verification.md").write_text("M1: no finding.\n")
    synthesize_qa(ctx)
    body = (tmp_path / QA_SYNTHESIS).read_text()
    assert "VERIFIER MISSING" not in body and "M1: no finding." in body


# ── limits: the node's envelope and the run's wall hold across the retry ───────────


def test_the_spine_reserves_every_attempt_and_one_overhead(tmp_path):
    dsh = Scripted(tmp_path / "dsh", [{"answer": "ok"}])
    state = {"graph": _graph(_node(timeout_s=100)).to_state(), "knobs": {}}
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path, dsh.client().context(),
                                     lowering.node_declarations(state))
    assert lowered.nodes["reviewer"].command_timeout_s == 100 + 200 + NODE_DISPATCH_OVERHEAD_S
    spec = json.loads(Path(lowered.nodes["reviewer"].dispatch_command.rsplit(" ", 1)[1].strip("'")).read_text())
    assert spec["retry"] == RULE
    assert lowering.node_declarations(state)["reviewer"]["retry"] == RULE
    # Without a rule the reservation is what it was.
    state = {"graph": _graph(_node(timeout_s=100, retry=None)).to_state(), "knobs": {}}
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path, dsh.client().context(),
                                     lowering.node_declarations(state))
    assert lowered.nodes["reviewer"].command_timeout_s == 100 + NODE_DISPATCH_OVERHEAD_S
    assert lowering.node_declarations(state)["reviewer"].get("retry") is None


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_the_run_wall_shortens_the_retry(tmp_path, runtime):
    from refactory_dsh.budgets import ResourceLimits, RunBudget

    workdir = tmp_path / "run"
    workdir.mkdir()
    budget = RunBudget.create(workdir, ResourceLimits(max_elapsed_s=40))
    budget.resume()
    # Streams for 4 s, then goes silent: it times out after about 5 s.
    dsh = Scripted(tmp_path / "dsh", [{"active": 4, "sleep": 60}, {"answer": "in time"}])
    # An 8 s wall, then 80 s: the retry gets what is left of the run's 40 s instead.
    ok, error = _run(runtime, workdir, _graph(_node(retry={**RULE, "timeout_factor": 10}, timeout_s=8)),
                     dsh.client(), budget=budget)
    assert ok, error
    [event] = _retried(workdir)
    assert 26 <= event["wall_s"] <= 35


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_no_retry_starts_when_the_run_wall_is_spent(tmp_path, runtime):
    from refactory_dsh.budgets import ResourceLimits, RunBudget

    workdir = tmp_path / "run"
    workdir.mkdir()
    budget = RunBudget.create(workdir, ResourceLimits(max_elapsed_s=6))
    budget.resume()
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 60}, {"answer": "never asked"}])
    ok, error = _run(runtime, workdir, _graph(_node(timeout_s=4)), dsh.client(), budget=budget)
    assert not ok
    assert "its declared retry was not started" in error
    assert dsh.calls == 1 and _retried(workdir) == []


class _Slow:
    """A client that reports no limits: each call takes ``seconds`` and times out."""

    def __init__(self, seconds: float):
        self.seconds, self.calls = seconds, 0

    def complete(self, prompt, *, system="", workdir=None):
        self.calls += 1
        time.sleep(self.seconds)
        raise CompletionTimeout(f"model completion timed out after {self.seconds}s")


def test_the_retry_stays_inside_the_nodes_envelope(tmp_path, monkeypatch):
    # Walls of 1 s and 2 s reserve 3 s (with no slack here); a first session that overran
    # to 2.6 s is charged 3 s and leaves nothing, so the retry is not started rather than outliving
    # the envelope.
    from refactory_dsh import node_retry

    monkeypatch.setattr(node_retry, "ENVELOPE_SLACK_S", 0)
    client = _Slow(2.6)
    with pytest.raises(RetriesExhausted, match="its declared retry was not started"):
        deliver(tmp_path, "reviewer", prompt="p", writes=[REPORT], reads=[], client=client,
                ask=lambda c, p, place: c.complete(p), capture_final=True, rule=RULE, timeout_s=1)
    assert client.calls == 1


def test_a_budget_spent_during_the_first_session_stops_the_retry(tmp_path):
    from refactory_dsh.budgets import BudgetExhausted

    class Spent:
        def remaining_s(self):
            return 1000

        def checkpoint(self):
            raise BudgetExhausted("token budget exhausted")

    client = _Slow(0)
    with pytest.raises(BudgetExhausted):
        deliver(tmp_path, "reviewer", prompt="p", writes=[REPORT], reads=[], client=client,
                ask=lambda c, p, place: c.complete(p), capture_final=True, rule=RULE, timeout_s=100,
                budget=Spent())
    assert client.calls == 1
    assert _retried(tmp_path) == []


def test_an_attempt_that_changed_the_target_is_refused_not_retried(tmp_path, monkeypatch):
    """The target guard judges a timed-out attempt before any retry: its refusal stands."""
    from refactory_dsh import agent_delivery, target_guard

    monkeypatch.setattr(target_guard, "strayed", lambda workdir, before: (["tool.py"], []))
    refused = []

    def refuse(workdir, node, **kwargs):
        refused.append(node)
        raise ValueError("agent 'reviewer' changed the target")

    monkeypatch.setattr(agent_delivery, "finish", refuse)
    client = _Slow(0)
    with pytest.raises(ValueError, match="changed the target"):
        deliver(tmp_path, "reviewer", prompt="p", writes=[REPORT], reads=[], client=client,
                ask=lambda c, p, place: c.complete(p), capture_final=True, rule=RULE, timeout_s=100)
    assert refused == ["reviewer"] and client.calls == 1


def test_a_retry_session_may_run_past_the_routes_envelope(monkeypatch, tmp_path):
    """A retry's longer limit is its session's limit even above the route's envelope,
    which a node's own bound only ever lowers, on a backend that reports no activity.
    On one that does, there is no route envelope: a declared wall stands as declared."""
    seen = []

    def capture(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        seen.append(budget_s)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setenv("REFACTORY_MODEL_SCRATCH", str(tmp_path))
    monkeypatch.setattr(DshModelClient, "_run_completion", capture)
    client = DshModelClient(worker_backend="codex", model="m").for_node("reviewer")
    client.with_wall(10 ** 6).complete("hello", workdir=tmp_path)
    client.with_retry_wall(10 ** 6).complete("hello", workdir=tmp_path)
    assert seen[0] < 10 ** 6 and seen[1] == 10 ** 6
    seen.clear()
    client = DshModelClient(provider="p", model="m").for_node("reviewer")
    client.with_wall(10 ** 6).complete("hello", workdir=tmp_path)
    client.with_retry_wall(10 ** 6).complete("hello", workdir=tmp_path)
    assert seen == [10 ** 6, 10 ** 6]


# ── the declaration and the shared-state rule ─────────────────────────────────────


def test_the_declaration_has_one_canonical_form():
    assert retry_declaration({}) == RULE
    assert retry_declaration({"timeout_factor": 2.0}) == RULE
    assert retry_declaration(None) is None
    # times 0 declares no retry, turning off a data pass's default.
    assert retry_declaration({"times": 0})["times"] == 0
    for bad, says in [({"on": ["error"]}, "unknown"), ({"on": "timeout"}, "non-empty list"),
                      ({"times": -1}, "times"), ({"times": 5}, "times"), ({"times": True}, "times"),
                      ({"timeout_factor": 0.5}, "timeout_factor"), ({"wait": 1}, "unknown"),
                      ("timeout", "mapping")]:
        with pytest.raises(ValueError, match=says):
            retry_declaration(bad)


def test_the_declaration_survives_the_state_round_trip_and_is_absent_by_default():
    node = _node()
    assert Node.from_state(node.to_state()).retry == RULE
    assert Node.from_state(node.to_state()).outputs_only is True
    # Unset, the state is what it was, so recorded candidate ids hold.
    plain = Node("y", "agent", prompt_template="p").to_state()
    assert "retry" not in plain and "outputs_only" not in plain
    workflow = lowering.to_workflow({"graph": _graph(Node("y", "agent", prompt_template="p")).to_state()})
    assert "retry" not in workflow.to_dict()["nodes"]["y"]
    assert lowering.to_workflow({"graph": _graph(node).to_state()}).to_dict()["nodes"]["reviewer"]["retry"] == RULE


def test_a_node_that_may_change_shared_state_cannot_declare_a_retry():
    with pytest.raises(GraphError, match="may change shared state"):
        Node("emit", "agent", prompt_template="Build it.", writes=("emit.txt",), timeout_s=60,
             retry=RULE)
    with pytest.raises(GraphError, match="not an agent"):
        Node("x", "op", call="m:f", retry=RULE)
    with pytest.raises(GraphError, match="no writes"):
        Node("x", "agent", prompt_template="p", outputs_only=True)
    with pytest.raises(GraphError, match="reads what it writes"):
        Node("x", "agent", prompt_template="p", reads=("design.md",), writes=("design.md",),
             outputs_only=True)
    with pytest.raises(GraphError, match="no timeout_s"):
        Node("x", "agent", prompt_template="p", writes=("r.md",), outputs_only=True, retry=RULE)
    # An edit that adds a retry to the builder is refused the same way.
    with pytest.raises(GraphError, match="may change shared state"):
        Node("emit", "agent", prompt_template="p", writes=("emit.txt",),
             timeout_s=60).with_field("retry", RULE)
    # A node with a clone workspace is safe to retry: each attempt gets a fresh clone.
    # A clone and a scratch folder both keep writes inside the workspace until collect,
    # and collect runs only after the session returns, so a timed-out attempt contributes nothing.
    Node("emit", "agent", prompt_template="Build it.", writes=("emit.txt",),
         timeout_s=60, retry=RULE,
         workspace={"kind": "clone", "of": "attempt", "collect": "diff"})
    # A node with a scratch workspace is equally safe: the scratch folder is discarded
    # on retry just as a clone is, and collect gates the writes the same way.
    Node("scratch_node", "agent", prompt_template="Scratch it.", writes=("out.txt",),
         timeout_s=60, retry=RULE,
         workspace={"kind": "scratch", "collect": "files"})


@pytest.mark.parametrize("workspace", [{"kind": "none"}, {}, "none"])
def test_a_workspace_that_means_none_is_not_isolation(workspace):
    """The guard reads the workspace in canonical form: a declaration that means "no
    workspace" is not a fresh attempt, so a retry on it is refused like any other node
    that may change shared state. Canonicalising below the guard instead lets the raw
    dict through and the retry is wrongly accepted."""
    with pytest.raises(GraphError, match="may change shared state"):
        Node("emit", "agent", prompt_template="Build it.", writes=("emit.txt",),
             timeout_s=60, retry=RULE, workspace=workspace)


def test_design_mode_retries_its_builder_and_verifier():
    from refactory_dsh.design import BUILDER_RETRY, BUILDER_TIMEOUT_S, DesignMode, TIMEOUT_RETRY

    graph = DesignMode().build()
    assert [node.id for node in graph.nodes.values() if node.retry is not None] == ["emit", "verifier"]
    assert sorted(node.id for node in graph.nodes.values() if node.outputs_only) == [
        "qa_director_rerun", "verifier"]
    assert graph.nodes["verifier"].retry == TIMEOUT_RETRY == RULE
    # The builder's limit is a hard maximum (inactivity stops a stuck one sooner), and its
    # retry has the same maximum rather than twice it.
    assert graph.nodes["emit"].retry == BUILDER_RETRY
    assert graph.nodes["emit"].timeout_s == BUILDER_TIMEOUT_S == 14400


def test_other_modes_declare_no_retry():
    from test_lowering_smoke import _mode_classes

    for key, mode_cls in _mode_classes():
        if key == "default_pack-design":
            continue
        state = mode_cls().state_dict()
        nodes = state["graph"]["nodes"]
        if key == "spreadsheetbench-solver":
            # SkillOpt's 5 attempts per call (4 retries), on the solve node only; a
            # crashing solve run through them: test_spreadsheetbench's
            # test_a_crashed_solve_call_gets_five_attempts.
            assert {k: n["retry"] for k, n in nodes.items() if "retry" in n} == \
                {"solve": {"on": ["crash"], "times": 4, "timeout_factor": 2}}
            continue
        if key in ("spreadsheetbench-agent", "spreadsheetbench_v2-agent"):
            # SkillOpt's one retry on an empty response, on the solve node only:
            # test_spreadsheetbench_agent's test_the_command_line_is_skillopts.
            assert {k: n["retry"] for k, n in nodes.items() if "retry" in n} == \
                {"solve": {"on": ["crash"], "times": 1, "timeout_factor": 2}}
            continue
        assert all("retry" not in node for node in nodes.values()), key


# ── builder retry (workspace-based isolation) ────────────────────────────────────

EMIT_FILE = "emit.txt"
BUILDER_WORKSPACE = {"kind": "clone", "of": "attempt", "collect": "diff"}


def _builder_node(**overrides) -> Node:
    defaults = dict(prompt_template="Build it.", writes=(EMIT_FILE,), timeout_s=2, retry=RULE,
                    workspace=dict(BUILDER_WORKSPACE))
    defaults.update(overrides)
    return Node("emit", "agent", **defaults)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_builder_that_times_out_once_is_retried_and_the_run_continues(tmp_path, runtime, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKSPACE_SANDBOX", "unconfined")
    dsh = Scripted(tmp_path / "dsh", [
        {"sleep": 30, "write": {EMIT_FILE: "partial build"}},
        {"write": {EMIT_FILE: "complete build"}, "answer": "done"},
    ])
    workdir = tmp_path / "run"
    if runtime == "spine":
        from refactory_dsh import worktree
        from tests_support import git_repo
        workdir.mkdir()
        worktree.create(git_repo(tmp_path / "repo"), "HEAD", "0123456789ab", workdir)
    ok, error = _run(runtime, workdir, _graph(_builder_node()), dsh.client())
    assert ok, error
    assert dsh.calls == 2
    assert "complete build" in (workdir / EMIT_FILE).read_text()
    [event] = _retried(workdir)
    assert (event["node_id"], event["attempt"], event["attempts"], event["wall_s"]) == ("emit", 2, 2, 4)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_builders_retry_workspace_holds_none_of_the_first_attempts_edits(tmp_path, runtime, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKSPACE_SANDBOX", "unconfined")
    dsh = Scripted(tmp_path / "dsh", [
        {"sleep": 30, "write": {EMIT_FILE: "partial", "stale.txt": "first attempt"}},
        {"fresh": True, "write": {EMIT_FILE: "clean build"}, "answer": "done"},
    ])
    workdir = tmp_path / "run"
    if runtime == "spine":
        from refactory_dsh import worktree
        from tests_support import git_repo
        workdir.mkdir()
        worktree.create(git_repo(tmp_path / "repo"), "HEAD", "0123456789ab", workdir)
    ok, error = _run(runtime, workdir, _graph(_builder_node()), dsh.client())
    assert ok, error
    assert dsh.calls == 2
    assert not (workdir / "stale.txt").exists()
    assert "clean build" in (workdir / EMIT_FILE).read_text()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_builder_that_times_out_twice_stops_with_the_retry_spent_error_naming_both_walls(
        tmp_path, runtime, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKSPACE_SANDBOX", "unconfined")
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30}])
    workdir = tmp_path / "run"
    if runtime == "spine":
        from refactory_dsh import worktree
        from tests_support import git_repo
        workdir.mkdir()
        worktree.create(git_repo(tmp_path / "repo"), "HEAD", "0123456789ab", workdir)
    ok, error = _run(runtime, workdir, _graph(_builder_node(timeout_s=3)), dsh.client())
    assert not ok
    assert re.search(r"emit timed out twice \(after [12]s, then [12]s\)", error), error
    assert dsh.calls == 2


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_scratch_declared_node_may_also_retry(tmp_path, runtime, monkeypatch):
    monkeypatch.setenv("REFACTORY_WORKSPACE_SANDBOX", "unconfined")
    dsh = Scripted(tmp_path / "dsh", [
        {"sleep": 30, "write": {"out.txt": "partial"}},
        {"write": {"out.txt": "scratch done"}, "answer": "done"},
    ])
    workdir = tmp_path / "run"
    node = Node("scratch_node", "agent", prompt_template="Scratch it.", writes=("out.txt",),
                timeout_s=2, retry=RULE,
                workspace={"kind": "scratch", "collect": "files"})
    ok, error = _run(runtime, workdir, _graph(node), dsh.client())
    assert ok, error
    assert dsh.calls == 2
    assert (workdir / "out.txt").read_text().strip() == "scratch done"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_retry_cannot_outrun_the_run_wall(tmp_path, runtime, monkeypatch):
    from refactory_dsh.budgets import ResourceLimits, RunBudget

    monkeypatch.setenv("REFACTORY_WORKSPACE_SANDBOX", "unconfined")
    workdir = tmp_path / "run"
    workdir.mkdir()
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 60}, {"answer": "never asked"}])
    if runtime == "spine":
        from refactory_dsh import worktree
        from tests_support import git_repo
        worktree.create(git_repo(tmp_path / "repo"), "HEAD", "0123456789ab", workdir)
    # Budget of 30 s, started after the test's own setup: whenever the first attempt goes
    # silent, less than 30 s is left, below the 40 s the retry wants and below the shortest
    # retry worth starting (`MIN_RETRY_WALL_S`), so the retry is not started. The run's
    # start-up (slow on a loaded machine, or on macOS) has nearly all of the 30 s to spend
    # before the budget itself runs out.
    budget = RunBudget.create(workdir, ResourceLimits(max_elapsed_s=30))
    budget.resume()
    ok, error = _run(runtime, workdir, _graph(_builder_node(timeout_s=20)), dsh.client(), budget=budget)
    assert not ok
    assert "its declared retry was not started" in error
    assert dsh.calls == 1 and _retried(workdir) == []


def test_the_builder_knob_sets_its_timeout_and_both_runtimes_use_it(tmp_path):
    from refactory_dsh.design import BUILDER_RETRY as TIMEOUT_RETRY, DesignMode
    from refactory_dsh.workspace import preparation_budget_s

    # A person's knob, exactly as the run path applies it: the graph is built from the
    # defaults and the state dict carries the override into both runtimes.
    from refactory_dsh.interactive import with_knobs
    mode = DesignMode()
    loaded = with_knobs(mode.state_dict(), ["builder_timeout_s=600"])
    mode.load_state_dict(loaded)
    assert mode.graph().nodes["emit"].timeout_s == 600
    assert Graph.from_state(mode.state_dict()["graph"]).nodes["emit"].timeout_s == 600
    graph = mode.graph()
    state = mode.state_dict()
    dsh = Scripted(tmp_path / "dsh", []).client()
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path / "run", dsh.context(),
                                     lowering.node_declarations(state))
    check_timeout = state.get("knobs", {}).get("check_timeout_s", {}).get("value")
    expected = (node_allowance_s(600, TIMEOUT_RETRY) + NODE_DISPATCH_OVERHEAD_S
                + 2 * preparation_budget_s(graph.nodes["emit"].workspace, check_timeout))
    assert lowered.nodes["emit"].command_timeout_s == expected


def test_walker_and_spine_give_the_builder_the_same_walls(tmp_path):
    from refactory_dsh.design import BUILDER_RETRY as TIMEOUT_RETRY, DesignMode
    from refactory_dsh.workspace import preparation_budget_s

    state = DesignMode().state_dict()
    graph = Graph.from_state(state["graph"])
    assert graph.nodes["emit"].timeout_s == 14400
    assert graph.nodes["emit"].retry == TIMEOUT_RETRY
    assert attempt_walls(14400, TIMEOUT_RETRY) == [14400, 14400]
    dsh = Scripted(tmp_path / "dsh", []).client()
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path / "run", dsh.context(),
                                     lowering.node_declarations(state))
    check_timeout = state.get("knobs", {}).get("check_timeout_s", {}).get("value")
    expected = (node_allowance_s(14400, TIMEOUT_RETRY) + NODE_DISPATCH_OVERHEAD_S
                + 2 * preparation_budget_s(graph.nodes["emit"].workspace, check_timeout))
    assert lowered.nodes["emit"].command_timeout_s == expected


# ── the pieces both runtimes share ───────────────────────────────────────────────


def test_walls_and_allowance():
    assert attempt_walls(1200, RULE) == [1200, 2400]
    assert attempt_walls(10, {"on": ["timeout"], "times": 2, "timeout_factor": 1.5}) == [10, 15, 23]
    assert node_allowance_s(1200, RULE) == 3600 and node_allowance_s(1200, None) == 1200


def test_only_a_timeout_is_a_timeout():
    from refactory_dsh.budgets import BudgetExhausted

    assert failure_kind(CompletionTimeout("model completion timed out after 1s")) == "timeout"
    assert failure_kind(RuntimeError("model completion timed out after 1s")) == "timeout"
    assert failure_kind(subprocess.TimeoutExpired("dsh", 1)) == "timeout"
    assert failure_kind(RuntimeError("model completion failed (exit 1): timed out reading")) is None
    assert failure_kind(BudgetExhausted("elapsed-time budget exhausted")) is None
    # A declared wall is the pipeline's own limit; a retry on timeout the node declares
    # answers it (both are the graph's choice). The backstop is the harness's guard: no
    # retry answers it.
    from refactory_dsh.model import DeclaredTimeout, HarnessBackstop
    assert failure_kind(DeclaredTimeout("model session timed out at its node's declared timeout_s")) == "timeout"
    assert failure_kind(HarnessBackstop("model session hit the harness backstop after 9s")) is None


def test_status_shows_the_retries(tmp_path):
    from refactory_dsh import readable
    from refactory_dsh.live import emit

    emit(tmp_path, RETRY_EVENT, node_id="verifier", attempt=2)
    emit(tmp_path, "node_started", node_id="verifier")
    assert retries(tmp_path) == {"verifier": 1}
    text = readable.format_status({"authoring": {"completed": []}, "state": "running",
                                   "activity": {"node_retries": retries(tmp_path)}})
    assert "Retried after a timeout: verifier (1x)" in text
    from refactory_dsh.console import status

    assert status(tmp_path)["activity"]["node_retries"] == {"verifier": 1}


def test_a_session_that_ignores_its_shutdown_is_not_retried(tmp_path, monkeypatch):
    """A session that does not stop within the grace is a forced exit with cleanup
    unconfirmed (`owned_process.shutdown`): its tools may still be writing, so it is
    never retried, however its deadline started."""
    from refactory_dsh import owned_process

    monkeypatch.setattr(owned_process, "SHUTDOWN_GRACE_S", 1)
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30, "ignore_term": True}, {"answer": "fine"}])
    ok, error = _run("walker", tmp_path / "run", _graph(_node(timeout_s=3)), dsh.client())
    assert not ok and "did not finish shutdown" in error
    assert dsh.calls == 1 and _retried(tmp_path / "run") == []


def _watched_run(tmp_path: Path) -> tuple[Path, Path]:
    """A run whose baseline watches a target and the run's own worktree, as prepare
    records it for a design run in a worktree."""
    from refactory_dsh import target_guard

    target, worktree, workdir = tmp_path / "target", tmp_path / "worktree", tmp_path / "run"
    for folder in (target, worktree, workdir):
        folder.mkdir()
    (worktree / "tool.py").write_text("VERSION = 2\n")
    target_guard.record(workdir, target, output_only=("emit",), also=(worktree,))
    return workdir, worktree


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_verifier_still_delivers_while_a_sibling_writes_into_the_worktree(tmp_path, runtime):
    """The review's repro, in the canary's shape: the verifier runs in fork_qa beside the
    check and the reviewers, which write into the run's worktree. A verifier that writes
    only its report delivers it whatever they write there while it runs."""
    workdir, worktree = _watched_run(tmp_path)
    # The scripted session writes the sibling's file itself, while the verifier runs.
    dsh = Scripted(tmp_path / "dsh", [{"answer": "M1: no finding.",
                                       "write": {str(worktree / "probe_output.txt"): "sibling\n"}}])
    ok, error = _run(runtime, workdir, _graph(_node()), dsh.client())
    assert ok, error
    assert (workdir / REPORT).read_text().strip() == "M1: no finding."
    assert (worktree / "probe_output.txt").exists()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_timed_out_attempt_that_changed_the_worktree_is_not_retried(tmp_path, runtime):
    """A retry would start from what the session (or a sibling) left in the run's worktree,
    so a timed-out attempt during which it changed is the node's failure: one call, no retry."""
    workdir, worktree = _watched_run(tmp_path)
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30, "write": {str(worktree / "probe.py"): "x = 1\n"}},
                                      {"answer": "never asked"}])
    ok, error = _run(runtime, workdir, _graph(_node()), dsh.client())
    assert not ok
    assert "not retried: the run's worktree changed while it ran" in error and "probe.py" in error
    assert dsh.calls == 1 and _retried(workdir) == []


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("kind", ["scratch", "clone"])
def test_each_retry_prepares_a_fresh_workspace_and_collects_its_outputs(tmp_path, runtime, kind, monkeypatch):
    # The first attempt leaves a report and another file in its workspace. Only a new
    # dispatched workspace lets the second session prove neither survived. Drive
    # both actual runtime entry points so a hook around the whole retry loop fails.
    monkeypatch.setenv("REFACTORY_WORKSPACE_SANDBOX", "unconfined")
    dsh = Scripted(tmp_path / "dsh", [
        {"sleep": 30, "write": {REPORT: "partial", "stale.txt": "first attempt"}},
        {"fresh": True, "write": {REPORT: "collected whole report"}, "answer": "summary"},
    ])
    workdir = tmp_path / "run"
    if kind == "clone":
        from refactory_dsh import worktree
        from tests_support import git_repo
        workdir.mkdir()
        worktree.create(git_repo(tmp_path / "repo"), "HEAD", "0123456789ab", workdir)
    ok, error = _run(runtime, workdir, _graph(_node(workspace=kind, timeout_s=3)), dsh.client())
    assert ok, error
    assert dsh.calls == 2
    assert (workdir / REPORT).read_text() == "collected whole report"
    assert not (workdir / "stale.txt").exists()
    assert len(_retried(workdir)) == 1
    receipts = [json.loads(p.read_text()) for p in
                (workdir / ".refactory/agent-delivery").glob("*/receipt.json")]
    assert sorted(r["status"] for r in receipts) == ["delivered", "superseded"]


@pytest.mark.parametrize("kind", ["scratch", "clone"])
def test_workspace_preparation_is_reserved_for_every_retry_attempt(tmp_path, kind):
    from refactory_dsh.workspace import preparation_budget_s

    node = _node(workspace=kind, timeout_s=10)
    graph = _graph(node)
    client = Scripted(tmp_path / "dsh", []).client()
    state = {"graph": graph.to_state(), "knobs": {}}
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path / "run", client.context(),
                                     lowering.node_declarations(state))
    assert lowered.nodes[node.id].command_timeout_s == (
        30 + NODE_DISPATCH_OVERHEAD_S + 2 * preparation_budget_s(node.workspace))


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_an_optional_branch_whose_declared_retry_is_spent_at_its_wall_is_declared_timeout(
        tmp_path, runtime):
    """An optional verifier that reaches its declared wall on every attempt (its retry
    answers the first, the second is spent) is missing as ``declared_timeout``, the
    pipeline's limit (`node_retry._cause`), never ``timeout``, which #684 reads as the
    harness's."""
    verifier = Node("verifier", "agent", prompt_template="Verify.",
                    writes=("reviews/verification.md",), outputs_only=True, timeout_s=2,
                    retry=RULE)
    info = _missing_verifier(runtime, tmp_path, verifier, [{"active": 30}, {"active": 30}])
    assert info["cause"] == "declared_timeout", info


# ── a step with no tools that writes out tool calls it never made (D14) ─────────────

#: A reply that "ran" a check its step has no tool for and reports a made-up result
#: (create benchmark H2: 1999997 = 19 x 105263).
INVENTED = ('<tool_call>{"name": "sympy", "arguments": {"code": "isprime(1999997)"}}</tool_call>\n'
            '<tool_result>True</tool_result>\n1999997 is prime.')
#: Text of `node_retry.NO_TOOLS_NOTE`: a scripted plan keyed by it answers only a retry
#: that was told.
TOLD = "This step has no tools"
CLEAN = "1999997 = 19 x 105263."
ROOT = Path(__file__).resolve().parents[2]


def _solver(**fields) -> Node:
    return Node("solver", "agent", **{"prompt_template": "Is 1999997 prime?", "writes": (REPORT,),
                                      "tool_policy": "none", **fields})


def _no_tools_run(runtime: str, tmp_path: Path, plan: dict[str, list[dict]], budget=None,
                  client: dict | None = None, **fields):
    dsh = Scripted(tmp_path / "dsh", plan)
    ok, error = _run(runtime, tmp_path / "run", _graph(_solver(**fields)),
                     dsh.client(runtime_root=str(ROOT), **(client or {})), budget=budget)
    calls = sum(json.loads(p.read_text()) for p in (tmp_path / "dsh").glob("calls.json*"))
    return ok, error, calls


def _lowered_timeout(tmp_path: Path, node: Node) -> int:
    state = {"graph": _graph(node).to_state(), "knobs": {}}
    client = Scripted(tmp_path / "dsh", []).client(runtime_root=str(ROOT))
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path / "run", client.context(),
                                     lowering.node_declarations(state))
    return lowered.nodes[node.id].command_timeout_s


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_tool_less_reply_with_invented_tool_calls_is_retried_once_told_so(tmp_path, runtime):
    ok, error, calls = _no_tools_run(runtime, tmp_path, {
        "": [{"answer": INVENTED}], TOLD: [{"answer": CLEAN}]})
    workdir = tmp_path / "run"
    assert ok, error
    assert calls == 2
    assert (workdir / REPORT).read_text().strip() == CLEAN
    [event] = _retried(workdir)
    assert (event["node_id"], event["reason"]) == ("solver", INVENTED_TOOL_CALLS)
    assert (event["attempt"], event["attempts"]) == (2, 2)
    assert "<tool_call>, <tool_result>" in event["error"]
    receipts = [json.loads(p.read_text()) for p in (workdir / ".refactory/agent-delivery").glob("*/receipt.json")]
    assert sorted(r["status"] for r in receipts) == ["delivered", "superseded"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_second_invented_reply_fails_the_step_with_the_reason(tmp_path, runtime):
    ok, error, calls = _no_tools_run(runtime, tmp_path, {"": [{"answer": INVENTED}]})
    assert not ok
    assert ("solver has no tools, but its reply wrote out tool calls or results "
            "(<tool_call>, <tool_result>) again after it was told so") in error, error
    assert calls == 2
    assert not (tmp_path / "run" / REPORT).exists(), "the invented reply was delivered"


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_step_with_tools_keeps_a_reply_that_writes_out_tool_calls(tmp_path, runtime):
    ok, error, calls = _no_tools_run(runtime, tmp_path, {"": [{"answer": INVENTED}]},
                                     tool_policy=None)
    assert ok, error
    assert calls == 1
    assert (tmp_path / "run" / REPORT).read_text().strip() == INVENTED
    assert _retried(tmp_path / "run") == []


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_step_that_keeps_tool_markup_takes_the_reply_as_it_is(tmp_path, runtime):
    """LiveMath's steps reproduce SkillOpt's harness, which takes whatever the model wrote."""
    ok, error, calls = _no_tools_run(runtime, tmp_path, {"": [{"answer": INVENTED}]},
                                     keeps_tool_markup=True)
    assert ok, error
    assert calls == 1
    assert (tmp_path / "run" / REPORT).read_text().strip() == INVENTED
    assert _retried(tmp_path / "run") == []


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("shown", [
    f"The call it would make:\n```xml\n{INVENTED}\n```\nUnchecked here.",
    # A verdict quoting a person's feedback (`design.classify_reloop`): JSON as a whole.
    json.dumps({"verdict": "document",
                "reason": "they pasted <invoke name='python'>isprime(1999997)</invoke> here"}),
], ids=["fenced", "json-reply"])
def test_a_reply_that_only_shows_tool_call_markup_is_kept(tmp_path, runtime, shown):
    ok, error, calls = _no_tools_run(runtime, tmp_path, {"": [{"answer": shown}]})
    assert ok, error
    assert calls == 1
    assert (tmp_path / "run" / REPORT).read_text().strip() == shown


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_the_told_retry_and_the_nodes_own_rule_share_one_count(tmp_path, runtime):
    """Markup first; the told session goes silent and the node's own timeout rule retries it,
    still told; it answers. The events number every session the node may run."""
    ok, error, calls = _no_tools_run(runtime, tmp_path, {
        "": [{"answer": INVENTED}], TOLD: [{"sleep": 30}, {"answer": CLEAN}]},
        timeout_s=2, retry=RULE)
    assert ok, error
    assert calls == 3
    assert (tmp_path / "run" / REPORT).read_text().strip() == CLEAN
    events = _retried(tmp_path / "run")
    assert [(e["reason"], e["attempt"], e["attempts"]) for e in events] == [
        (INVENTED_TOOL_CALLS, 2, 3), ("timeout", 3, 3)]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_slow_invented_reply_is_retried_under_its_wall(tmp_path, runtime):
    """A tool-less step with its own timeout: the first reply takes 2 s and writes out a
    call; the retry runs under the same 3 s wall, on both runtimes."""
    ok, error, calls = _no_tools_run(runtime, tmp_path, {
        "": [{"sleep": 2, "answer": INVENTED}], TOLD: [{"answer": CLEAN}]},
        client={"inactivity_s": 10}, timeout_s=3)
    assert ok, error
    assert calls == 2
    [event] = _retried(tmp_path / "run")
    assert (event["reason"], event["wall_s"], event["after_s"]) == (INVENTED_TOOL_CALLS, 3, 2)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_no_told_retry_starts_when_the_run_wall_is_spent(tmp_path, runtime):
    from refactory_dsh.budgets import ResourceLimits, RunBudget

    workdir = tmp_path / "run"
    workdir.mkdir()
    budget = RunBudget.create(workdir, ResourceLimits(max_elapsed_s=4))
    budget.resume()
    ok, error, calls = _no_tools_run(runtime, tmp_path, {
        "": [{"sleep": 2, "answer": INVENTED}], TOLD: [{"answer": CLEAN}]},
        budget=budget, client={"inactivity_s": 10}, timeout_s=3)
    assert not ok
    assert "solver has no tools, but its reply wrote out tool calls or results" in error, error
    assert "its retry was not started" in error
    assert calls == 1 and _retried(workdir) == []


def test_the_spine_reserves_the_told_retry(tmp_path):
    """The extra session's wall and overhead, in the one dispatch the spine starts."""
    one = 100 + NODE_DISPATCH_OVERHEAD_S
    assert _lowered_timeout(tmp_path, _solver(timeout_s=100)) == 2 * one
    assert _lowered_timeout(tmp_path, _solver(timeout_s=100, keeps_tool_markup=True)) == one
    assert _lowered_timeout(tmp_path, _solver(timeout_s=100, tool_policy=None)) == one
    # With a rule: every attempt, plus one more as long as the longest.
    assert _lowered_timeout(tmp_path, _solver(timeout_s=100, retry=RULE)) == (
        100 + 200 + 200 + 2 * NODE_DISPATCH_OVERHEAD_S)


def test_livemath_and_the_skill_graph_keep_their_replies_as_skillopt_does():
    from refactory_packages.livemath import blocks, skill_graph

    builds = (blocks.build_baseline, blocks.build_solve_then_verify,
              blocks.build_two_attempts_then_judge, blocks.build_solve_then_self_check)
    nodes = [*(node for build in builds for node in build().nodes),
             *skill_graph.SkillGraph().build().nodes.values()]
    agents = [node for node in nodes if node.kind == "agent"]
    assert agents and all(node.tool_policy == "none" and node.keeps_tool_markup for node in agents)


@pytest.mark.parametrize("edit, copies", [
    ({"op": "insert", "after": "solve"}, ["solve_next"]),
    ({"op": "parallelize", "steps": ["solve"], "copies": 2}, ["solve_copy1", "solve_copy2"]),
], ids=["insert", "parallelize"])
def test_a_copy_of_livemaths_solve_keeps_its_replies_as_solve_does(edit, copies):
    """A copy is sent what the step it copies is, and takes its reply the same way."""
    from refactory_dsh.nn import edits as nn_edits
    from refactory_packages.livemath.mode import Solver

    graph = Graph.from_state(nn_edits.apply(edit, Solver().state_dict())["graph"])
    assert graph.nodes["solve"].keeps_tool_markup
    assert all(graph.nodes[copy].tool_policy == "none" and graph.nodes[copy].keeps_tool_markup
               for copy in copies), edit


def test_a_split_copy_keeps_its_replies_and_the_new_reviewer_refuses():
    """LiveMath's solve alone reads frozen knobs, so split refuses it; on a step that keeps
    its replies, split's copy keeps them and its new reviewer takes the platform's refusal."""
    import structure_fuzz as sf
    from refactory_dsh.nn import edits as nn_edits
    from refactory_packages.livemath.mode import Solver

    with pytest.raises(nn_edits.EditError, match="frozen knob"):
        nn_edits.apply({"op": "split", "step": "solve"}, Solver().state_dict())
    state = sf.seeds()["fuzz_draft_review"].state()
    state["graph"]["nodes"]["review"]["keeps_tool_markup"] = True
    graph = Graph.from_state(nn_edits.apply({"op": "split", "step": "review"}, state)["graph"])
    assert graph.nodes["review_a"].tool_policy == "none" and graph.nodes["review_a"].keeps_tool_markup
    assert not graph.nodes["review_b"].keeps_tool_markup
