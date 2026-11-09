"""A crashed agent step in a data pass is retried, and a spent one is missing, not failed.

On Countdown, about a third of Haiku 4.5's solves ran into Claude Code's output token cap
and came back as a product failure (``category: invalid-result``); the failure ended the
item, so a check-and-retry loop or a parallel fork could not recover it. Now:

- an agent step in a data pass gets `PASS_RETRY_TIMES` fresh sessions on a crash; after
  the output limit the retry is told so;
- when those crash too, the step is missing (and its item's record says so): a gate after it sees no answer and may
  reloop, a fork's join sees the branch without one, and the item is scored;
- outside a data pass nothing changes unless the node declares a retry.

Each runtime runs the same graph against a scripted ``dsh`` through the real model
client: the walker (`nn.executor`) in process, the spine through its real drain, whose
dispatched workers (`worker_node`) run the retry (`node_retry.deliver`).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, nn, node_retry, workers
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import graph as nn_graph
from refactory_dsh.nn.executor import CURRENT_ITEM, execute
from refactory_dsh.node_retry import RETRY_EVENT

# Read with defaults, so each test fails on a build without the rule for what it checks,
# not on the import.
PASS_RETRY_TIMES = getattr(nn_graph, "PASS_RETRY_TIMES", 2)
OUTPUT_LIMIT_NOTE = getattr(node_retry, "OUTPUT_LIMIT_NOTE", "ran out of output room")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NODE = shutil.which("node")
DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
PYTHON_BIN = str(REPO_ROOT / ".venv/bin")

needs_spine = pytest.mark.skipif(
    NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)

TASK = "loop_ops:ExhaustTask"  # items 0-3, scored from item_result.json, else answer.txt

#: What the Claude Code adapter now reports for a response cut off at the output cap.
OUTPUT_LIMIT = ("dsh: worker error: Product subagent failure (product: Claude Code; stage: "
                "query-run; category: output-limit; exit code: 1): output limit reached after "
                "128000 output tokens: each response is capped at 32000 tokens and the model "
                "had not answered when Claude Code stopped resuming it")
#: A product failure that is not the output limit.
WORKER_CRASH = ("dsh: worker error: Product subagent failure (product: Claude Code; stage: "
                "query-run; category: invalid-result; exit code: 1): the answer was empty")

FAKE_DSH = '''import fcntl, json, pathlib, sys
root = pathlib.Path({root!r})
plans = json.loads((root / "plan.json").read_text())
prompt = sys.stdin.read()
key = next((key for key in plans if key and key in prompt), "")
with open(root / "lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    counter = root / ("calls." + ("".join(c for c in key if c.isalnum()) or "all"))
    calls = json.loads(counter.read_text()) if counter.exists() else 0
    counter.write_text(json.dumps(calls + 1))
    with open(root / "prompts.jsonl", "a") as log:
        log.write(json.dumps({{"key": key, "prompt": prompt}}) + "\\n")
plan = plans[key]
step = plan[calls] if calls < len(plan) else plan[-1]
if step.startswith("dsh:"):
    print(step, file=sys.stderr)
    sys.exit(1)
print(step)
'''


class Scripted:
    """A scripted ``dsh``: each prompt key's answers in order (the last repeats); an
    answer starting ``dsh:`` is a crash with that diagnostic."""

    def __init__(self, root: Path, plan: dict[str, list[str]]):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        (root / "plan.json").write_text(json.dumps(plan))
        script = root / "fake_dsh.py"
        script.write_text(FAKE_DSH.format(root=str(root)))
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)

    def calls(self, key: str = "") -> int:
        path = self.root / ("calls." + ("".join(c for c in key if c.isalnum()) or "all"))
        return json.loads(path.read_text()) if path.exists() else 0

    def prompts(self, key: str = "") -> list[str]:
        path = self.root / "prompts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [row["prompt"] for row in rows if row["key"] == key]

    def client(self) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=60)


def _solver(node_id: str = "solve", prompt: str = "Solve.", writes: str = "answer.txt",
            **fields: Any) -> nn.Node:
    return nn.Node(node_id, "agent", prompt_template=prompt, writes=(writes,), tool_policy="none",
                   role="solver", **fields)


def _data(entry: str, exit_: str) -> nn.Node:
    return nn.Node("data", "data", subgraph_entry=entry, subgraph_exit=exit_,
                   writes=(CURRENT_ITEM,), produces=("item_scores",), task_ref=TASK)


def _straight(**solver: Any) -> nn.Graph:
    """data -> [solve] -> report: the solver is the whole pass."""
    graph = nn.Graph()
    graph.add(_data("solve", "solve"))
    graph.add(_solver(**solver))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("data", "report")
    return graph


def _looped() -> nn.Graph:
    """data -> [solve -> check (reloop solve | proceed score)] -> report."""
    graph = nn.Graph()
    graph.add(_data("solve", "score"))
    graph.add(_solver())
    graph.add(nn.Node("check", "gate", evaluator="crash_ops:check", reads=("answer.txt",),
                      max_iterations=2))
    graph.add(nn.Node("score", "op", call="crash_ops:score", reads=("answer.txt",),
                      writes=("item_result.json",)))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("solve", "check")
    graph.connect("check", "solve", condition="reloop")
    graph.connect("check", "score", condition="proceed")
    graph.connect("data", "report")
    return graph


def _forked() -> nn.Graph:
    """data -> [entry -> fork (a, b) -> join -> pick] -> report: two attempts, one pick."""
    graph = nn.Graph()
    graph.add(_data("entry", "pick"))
    graph.add(nn.Node("entry", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork", "fork", targets=("a", "b")))
    graph.add(_solver("a", "Angle A.", "a.txt"))
    graph.add(_solver("b", "Angle B.", "b.txt"))
    graph.add(nn.Node("join", "join", sources=("a", "b")))
    graph.add(nn.Node("pick", "op", call="crash_ops:pick", reads=("a.txt", "b.txt"),
                      writes=("item_result.json",)))
    graph.add(nn.Node("report", "op", call="loop_ops:noop"))
    graph.connect("entry", "fork")
    graph.connect("fork", "join")
    graph.connect("a", "join")
    graph.connect("b", "join")
    graph.connect("join", "pick")
    graph.connect("data", "report")
    return graph


def _walker(graph: nn.Graph, run: Path, dsh: Scripted) -> list[dict[str, Any]]:
    result = execute(graph, run, model=dsh.client(), max_steps=400)
    assert result.exit_ok and not result.halted, result.error
    return result.facts["item_scores"]


def _spine(graph: nn.Graph, run: Path, tmp_path: Path, dsh: Scripted) -> list[dict[str, Any]]:
    """Lower ``graph`` with dispatched workers, as a run on DSH does, and drive it through
    the real spine drain; return the run's item scores."""
    run.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state()}
    lowering.write_declarations(state, run)

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / f"specs-{run.name}", dsh.client().context(),
                                      lowering.node_declarations(state))

    graph_path, _ = lowering.write_mode_graph(state, tmp_path / f"graphs-{run.name}", name="crash",
                                              transform=dispatched)
    responses = tmp_path / f"responses-{run.name}.json"
    responses.write_text("{}")
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    env.update(REFACTORY_WORKDIR=str(run), PYTHONPATH=str(HERE),
               PATH=f"{PYTHON_BIN}:{env.get('PATH', '')}")
    done = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path), str(run),
         f"{PYTHON_BIN}/python", "60000", str(responses)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300, env=env)
    assert done.returncode == 0, f"stdout: {done.stdout[-2000:]}\nstderr: {done.stderr[-3000:]}"
    return json.loads((run / "facts.json").read_text())["item_scores"]


def _run(runtime: str, graph: nn.Graph, tmp_path: Path, dsh: Scripted) -> tuple[list, Path]:
    run = tmp_path / runtime
    if runtime == "walker":
        return _walker(graph, run, dsh), run
    return _spine(graph, run, tmp_path, dsh), run


def _keys(records: list[dict[str, Any]]) -> list[tuple[str, bool, float, Any]]:
    return sorted((str(r["id"]), bool(r["passed"]), float(r.get("score") or 0), r.get("error_kind"))
                  for r in records)


def _pass(run: Path, item: str) -> Path:
    [path] = list(run.glob(f"passes/data/{item}_*"))
    return path


def _events(directory: Path, kind: str) -> list[dict[str, Any]]:
    path = directory / ".refactory/events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("kind") == kind]


def _sessions(directory: Path) -> list[str]:
    """The state of every model session the pass started, from its worker receipts."""
    return sorted(json.loads(path.read_text())["state"]
                  for path in (directory / ".refactory/workers").glob("worker-*.json"))


RIGHT = [(str(i), True, 1.0, None) for i in range(4)]
RUNTIMES = ["walker", pytest.param("spine", marks=needs_spine)]


# ── a crash is retried, and the item is scored ───────────────────────────────


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_crashed_step_is_retried_and_its_item_scored(tmp_path, runtime):
    """[both runtimes] Item 0's first session hits the output limit; its retry, told it
    ran out of room, answers. Every item is scored, and the retry is recorded and counted
    (one worker receipt per session)."""
    dsh = Scripted(tmp_path / "dsh", {"": [OUTPUT_LIMIT, "right"]})
    scores, run = _run(runtime, _straight(), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls() == 5
    first, second, *_ = dsh.prompts()
    assert OUTPUT_LIMIT_NOTE not in first and OUTPUT_LIMIT_NOTE in second
    item = _pass(run, "0")
    [retried] = _events(item, RETRY_EVENT)
    assert retried["reason"] == "crash" and "output limit" in retried["summary"]
    assert "output limit reached after 128000 output tokens" in retried["error"]
    assert _sessions(item) == ["completed", "failed"]


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_step_whose_retries_all_crash_is_missing_and_its_item_still_scored(tmp_path, runtime):
    """[both runtimes] Every session of item 0 crashes: after `PASS_RETRY_TIMES` retries the
    step is missing, the item is graded without an answer (0, no error), and its siblings
    are scored. A product failure that is not the output limit adds no note."""
    dsh = Scripted(tmp_path / "dsh", {"": [WORKER_CRASH] * (1 + PASS_RETRY_TIMES) + ["right"]})
    scores, run = _run(runtime, _straight(), tmp_path, dsh)
    assert _keys(scores) == [("0", False, 0.0, None), *RIGHT[1:]]
    assert dsh.calls() == 1 + PASS_RETRY_TIMES + 3
    assert not any(OUTPUT_LIMIT_NOTE in prompt for prompt in dsh.prompts())
    item = _pass(run, "0")
    assert "solve crashed 3 times" in (item / "reviews/solve-missing.txt").read_text()
    [record] = [r for r in scores if str(r["id"]) == "0"]
    assert "solve crashed 3 times" in record["missing_steps"]["solve"]
    assert len(_events(item, RETRY_EVENT)) == PASS_RETRY_TIMES
    assert _sessions(item) == ["failed"] * (1 + PASS_RETRY_TIMES)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_crash_in_a_loop_is_an_attempt_its_gate_relooops_on(tmp_path, runtime):
    """[both runtimes] Item 0's solver crashes through its retries; the checker gate sees
    no answer and sends it back, and the next visit answers. The reloop's prompt carries the
    gate's reason and not the output-limit note."""
    dsh = Scripted(tmp_path / "dsh", {"": [OUTPUT_LIMIT] * (1 + PASS_RETRY_TIMES) + ["right"]})
    scores, run = _run(runtime, _looped(), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls() == 1 + PASS_RETRY_TIMES + 4
    reloop = dsh.prompts()[1 + PASS_RETRY_TIMES]
    assert "there is no answer" in reloop and OUTPUT_LIMIT_NOTE not in reloop
    # It answered on the reloop, so it is no longer missing.
    assert not (_pass(run, "0") / "reviews/solve-missing.txt").exists()
    assert all("missing_steps" not in record for record in scores)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_crashed_fork_branch_does_not_sink_its_sibling(tmp_path, runtime):
    """[both runtimes] Branch ``a`` crashes on every item, through its retries; branch ``b``
    answers. The join goes on with ``a`` missing, the pick sees one answer, and every item
    is scored."""
    dsh = Scripted(tmp_path / "dsh", {"Angle A.": [WORKER_CRASH], "Angle B.": ["right"], "": ["?"]})
    scores, run = _run(runtime, _forked(), tmp_path, dsh)
    assert _keys(scores) == RIGHT
    assert dsh.calls("Angle A.") == 4 * (1 + PASS_RETRY_TIMES)
    assert dsh.calls("Angle B.") == 4
    for item in "0123":
        assert json.loads((_pass(run, item) / "item_result.json").read_text())["answered"] == 1
        assert (_pass(run, item) / "reviews/a-missing.txt").exists()
    assert all(set(record["missing_steps"]) == {"a"} for record in scores)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_a_declared_times_zero_turns_the_default_off(tmp_path, runtime):
    """[both runtimes] ``retry: {"times": 0}`` restores the old behaviour: the crash is not
    retried and fails its item."""
    dsh = Scripted(tmp_path / "dsh", {"": [WORKER_CRASH, "right"]})
    scores, _ = _run(runtime, _straight(retry={"times": 0}), tmp_path, dsh)
    [(item, passed, score, kind), *rest] = _keys(scores)
    assert (item, passed, score) == ("0", False, 0.0) and kind is not None
    assert rest == RIGHT[1:]
    assert dsh.calls() == 4


# ── outside a data pass: unchanged ──────────────────────────────────────────


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_outside_a_data_pass_a_crash_is_not_retried(tmp_path, runtime):
    """[both runtimes] A crashed agent step that is not in a data pass fails as before:
    one session, no retry, no missing record. (A guard: it passes on main too.)"""
    graph = _unpassed()
    dsh = Scripted(tmp_path / "dsh", {"": [WORKER_CRASH, "right"]})
    run = tmp_path / runtime
    run.mkdir()
    if runtime == "walker":
        result = execute(graph, run, model=dsh.client(), max_steps=50)
        assert not result.exit_ok and "category: invalid-result" in (result.error or "")
    else:
        state = {"graph": graph.to_state()}
        lowering.write_declarations(state, run)
        lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path / "specs",
                                         dsh.client().context(), lowering.node_declarations(state))
        assert lowered.nodes["solve"].retry is None
        assert getattr(lowered.nodes["solve"], "missing_on_crash", None) is None
        env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
        done = subprocess.run(lowered.nodes["solve"].dispatch_command.split(), cwd=run, env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode != 0 and "category: invalid-result" in done.stderr
    assert dsh.calls() == 1
    assert _events(run, RETRY_EVENT) == []
    assert not (run / "reviews/solve-missing.txt").exists()


def _unpassed() -> nn.Graph:
    graph = nn.Graph()
    graph.add(_solver())
    graph.add(nn.Node("after", "op", call="loop_ops:noop"))
    graph.connect("solve", "after")
    return graph


# ── the rule, as the lowering hands it to the spine ─────────────────────────


def test_the_rule_resolves_from_the_graph_and_lowers_to_the_spine():
    assert _unpassed().retry_rule("solve") is None and not _unpassed().missing_on_crash("solve")
    graph = _straight()
    assert graph.retry_rule("solve") == {"on": ["crash"], "times": PASS_RETRY_TIMES, "timeout_factor": 1}
    # A timeout is not retried by default: the node's bound stays its author's.
    assert _straight(timeout_s=30).retry_rule("solve")["on"] == ["crash"]
    # A declared rule wins; times 0 is none.
    declared = _straight(retry={"on": ["crash"], "times": 1})
    assert declared.retry_rule("solve") == {"on": ["crash"], "times": 1, "timeout_factor": 2}
    assert _straight(retry={"times": 0}).retry_rule("solve") is None
    assert not _straight(retry={"times": 0}).missing_on_crash("solve")
    # A tool-using step that may change shared state gets no default.
    shared = nn.Graph()
    shared.add(_data("solve", "solve"))
    shared.add(nn.Node("solve", "agent", prompt_template="Solve.", writes=("answer.txt",)))
    assert shared.retry_rule("solve") is None and not shared.missing_on_crash("solve")

    lowered = lowering.to_workflow({"graph": graph.to_state()}).to_dict()["nodes"]["solve"]
    assert lowered["retry"] == {"on": ["crash"], "times": PASS_RETRY_TIMES, "timeout_factor": 1}
    assert lowered["missing_on_crash"] is True
    declarations = lowering.node_declarations({"graph": graph.to_state()})["solve"]
    assert declarations["missing_on_crash"] is True and declarations["retry"]["on"] == ["crash"]


def test_a_crash_retry_without_a_node_bound_reserves_every_session_on_the_spine(tmp_path):
    """The spine's command outlasts every session a crash retry may start."""
    from refactory_dsh.model import node_session_budget_s

    graph = _straight()
    state = {"graph": graph.to_state()}
    client = DshModelClient(worker_backend="dsh", provider="p", model="m", timeout_s=60, attempts=1)
    lowered = workers.lower_workflow(lowering.to_workflow(state), tmp_path, client.context(),
                                     lowering.node_declarations(state))
    one = node_session_budget_s(client.for_node("solve").with_tool_policy("none").retry_attempts(), 60)
    assert lowered.nodes["solve"].command_timeout_s == (1 + PASS_RETRY_TIMES) * one


def test_which_failures_are_crashes():
    from refactory_dsh.model import CompletionFailed, failure_kind as cause_kind
    from refactory_dsh.node_retry import failure_kind

    assert cause_kind(OUTPUT_LIMIT) == "output_limit"
    assert cause_kind("dsh: max-tokens: the model reached its output token limit") == "output_limit"
    assert cause_kind(WORKER_CRASH) == "worker"
    assert cause_kind("dsh: provider error: 401 authentication failed") == "setup"
    assert cause_kind("TRANSPORT: socket hang up") == "transport"
    assert failure_kind(CompletionFailed("model completion failed (exit 1): x", kind="worker")) == "crash"
    assert failure_kind(CompletionFailed("model completion failed (exit 1): x", kind="output_limit")) == "crash"
    assert failure_kind(CompletionFailed("model completion failed (exit 1): x", kind="setup")) is None
    # A bare status code inside a token count is not an HTTP 429.
    from refactory_dsh.model import is_transport_failure
    assert not is_transport_failure("output limit reached after 142903 output tokens")
    assert is_transport_failure("status 429 too many requests")


def test_the_optimizer_is_told_what_a_crash_does():
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim.base import composition_section

    state = _looped().to_state()
    nodes = [{"id": key, **value} for key, value in state["nodes"].items()]
    text = "\n".join(composition_section(nn_edits.catalogue(), nodes, state["edges"]))
    assert "**crashes**" in text and "fresh session" in text and "no answer" in text


# ── the outer loop: a flaky candidate is still scored on every item ─────────


class _Flaky:
    """A walker model whose every other session crashes the way the model client
    reports a product failure."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, **_: Any) -> str:
        from refactory_dsh import model

        self.calls += 1
        if self.calls % 2 == 0:
            crash = getattr(model, "CompletionFailed", RuntimeError)
            raise crash(f"model completion failed (exit 1): {WORKER_CRASH}")
        return "<answer>A</answer>"

    def for_node(self, node_id: str, role: str | None = None) -> "_Flaky":
        return self

    def with_tool_policy(self, policy: str) -> "_Flaky":
        return self

    def with_wall(self, seconds: Any) -> "_Flaky":
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


def test_an_outer_loop_evaluation_scores_every_item_of_a_flaky_candidate(tmp_path, monkeypatch):
    """[outer loop, walker] Half the solver's sessions crash; each crash is retried, so the
    evaluation records a score and no error for every item, and the crashed sessions are
    counted among the model calls."""
    from refactory_dsh.loop import Loop, LoopConfig

    from test_search_shakedown import _all_a_items

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(_all_a_items(tmp_path)))
    model = _Flaky()
    config = LoopConfig(epochs=1, batch_size=1, max_evaluations=1, optimizer="random",
                        seed=42, runs_root=tmp_path / "run", holdout=0.0)
    Loop("livemath/solver", config, project=Path("."), model=model).run()
    records = [record for trial in sorted((tmp_path / "run" / "runs").iterdir())
               for record in json.loads((trial / "item_records.json").read_text())]
    assert records
    assert [record["error_kind"] for record in records] == [None] * len(records)
    assert all(record["score"] is not None for record in records)
    # Every item answered once, and every crash before it was a session of its own.
    assert model.calls == 2 * len(records) - 1


def test_outside_data_passes_the_shipped_modes_are_unchanged():
    """Every shipped mode: a node outside a data pass runs exactly its declared rule (design
    mode's builder and verifier) or none, and is never missing on a crash; only agent steps
    in a data pass get the default."""
    from test_lowering_smoke import _mode_classes

    defaulted: dict[str, list[str]] = {}
    for key, mode_cls in _mode_classes():
        graph = nn.Graph.from_state(mode_cls().state_dict()["graph"])
        for node_id, node in graph.nodes.items():
            rule = graph.retry_rule(node_id)
            if graph.in_data_pass(node_id):
                if node.retry is None and rule is not None:
                    defaulted.setdefault(key, []).append(node_id)
                continue
            assert rule == node.retry, (key, node_id)
            assert not graph.missing_on_crash(node_id), (key, node_id)
    # The no-tools solvers of the outer-loop task modes get it.
    assert "livemath-solver" in defaulted
