"""Declared node recovery: a timed-out agent session is run once more, fresh and longer.

The rule is the graph's (`Node.retry`), and both runtimes run it through
`node_retry.deliver`: the walker from `Executor._visit_agent`, the spine from the worker
each dispatched agent node runs (`python -m refactory_dsh.worker_node <spec>`), inside the
envelope `workers.lower_workflow` reserves. The runtime tests here drive the real model
client against a scripted `dsh` that answers call by call, so a timeout is a real
deadline, not a raised string.
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
from refactory_dsh.node_retry import (RETRY_EVENT, RetriesExhausted, attempt_walls, deliver,
                                      failure_kind, node_allowance_s, retries)

RULE = {"on": ["timeout"], "times": 1, "timeout_factor": 2}
REPORT = "reviews/report.md"

FAKE_DSH = '''import json, sys, time, pathlib
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
                              attempts=1, timeout_s=60, **extra)


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
    # The first session writes part of its report and runs past its 2 s wall; the retry
    # takes 3 s, which only the doubled 4 s limit allows, and answers.
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30, "write": {REPORT: "half a report"}},
                                      {"sleep": 3, "answer": "the whole report"}])
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
    ok, error = _run(runtime, tmp_path / "run", _graph(_node(timeout_s=1)), dsh.client())
    assert not ok
    # The time each session actually ran: its limit, and a moment to shut down.
    assert re.search(r"reviewer timed out twice \(after [12]s, then [23]s\)", error), error
    assert dsh.calls == 2


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_without_a_rule_a_timeout_is_not_retried(tmp_path, runtime):
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 30}, {"answer": "late"}])
    ok, error = _run(runtime, tmp_path / "run", _graph(_node(timeout_s=1, retry=None)), dsh.client())
    assert not ok and "model completion timed out after 1s" in error
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
                    timeout_s=1, retry=RULE, outputs_only=True)
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
    dsh = Scripted(tmp_path / "dsh", [{"sleep": 60}, {"answer": "in time"}])
    # A 2 s wall, then 50 s: the retry gets what is left of the run's 40 s instead.
    ok, error = _run(runtime, workdir, _graph(_node(retry={**RULE, "timeout_factor": 10}, timeout_s=5)),
                     dsh.client(), budget=budget)
    assert ok, error
    [event] = _retried(workdir)
    assert 30 <= event["wall_s"] <= 35


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
    which a node's own bound only ever lowers."""
    seen = []

    def capture(self, cmd, run_dir, stop_at=None, budget_s=None, task=None, target_workdir=None):
        seen.append(budget_s)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setenv("REFACTORY_MODEL_SCRATCH", str(tmp_path))
    monkeypatch.setattr(DshModelClient, "_run_completion", capture)
    client = DshModelClient(provider="p", model="m").for_node("reviewer")
    client.with_wall(10 ** 6).complete("hello", workdir=tmp_path)
    client.with_retry_wall(10 ** 6).complete("hello", workdir=tmp_path)
    assert seen[0] < 10 ** 6 and seen[1] == 10 ** 6


# ── the declaration and the shared-state rule ─────────────────────────────────────


def test_the_declaration_has_one_canonical_form():
    assert retry_declaration({}) == RULE
    assert retry_declaration({"timeout_factor": 2.0}) == RULE
    assert retry_declaration(None) is None
    for bad, says in [({"on": ["error"]}, "unknown"), ({"on": "timeout"}, "non-empty list"),
                      ({"times": 0}, "times"), ({"times": 4}, "times"), ({"times": True}, "times"),
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


def test_design_mode_retries_only_its_verifier():
    # Only the node seen to time out declares a retry; a node is added when it is seen to.
    from refactory_dsh.design import DesignMode, TIMEOUT_RETRY

    graph = DesignMode().build()
    assert [node.id for node in graph.nodes.values() if node.retry is not None] == ["verifier"]
    assert [node.id for node in graph.nodes.values() if node.outputs_only] == ["verifier"]
    assert graph.nodes["verifier"].retry == TIMEOUT_RETRY == RULE


def test_other_modes_declare_no_retry():
    from test_lowering_smoke import _mode_classes

    for key, mode_cls in _mode_classes():
        if key == "default_pack-design":
            continue
        state = mode_cls().state_dict()
        assert all("retry" not in node for node in state["graph"]["nodes"].values()), key


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
    ok, error = _run("walker", tmp_path / "run", _graph(_node(timeout_s=1)), dsh.client())
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
    ok, error = _run(runtime, workdir, _graph(_node(workspace=kind, timeout_s=1)), dsh.client())
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
