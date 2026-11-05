"""Tests for the LiveMath solver-block library, dual-runtime equivalence, and grid CLI.

Coverage:
- Block construction and graph validation
- Baseline byte-exactness (graph state and SkillOpt prompt parity)
- Every agent node's route settings (tool_policy, timeout_s, system_prompt, skill)
- Walker integration for each block at parallelism 1 and 3
- Spine+walker equivalence for each block (gated by @needs_spine)
- Failed-branch and failed-solve scoring (F4: item scores 0, run continues)
- Grid command output with shortcut split
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

from refactory_dsh import lowering, nn
from refactory_dsh.nn.executor import execute

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "livemath"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NODE = shutil.which("node")
DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
PYTHON_BIN = str(REPO_ROOT / ".venv/bin")

needs_spine = pytest.mark.skipif(
    NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)


# ── helpers ───────────────────────────────────────────────────────────────


def _patch_task(monkeypatch):
    """Patch LiveMathTask to use fixture data with no shuffling."""
    import refactory_packages.livemath.tasks as tasks_mod

    original_init = tasks_mod.LiveMathTask.__init__

    def patched_init(self, data_path=None, shuffle_choices=True):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False)

    monkeypatch.setattr(tasks_mod.LiveMathTask, "__init__", patched_init)


class _NodeScriptedModel:
    """A model that returns different answers based on the node id.

    Accepts a dict mapping node_id -> answer text. Falls back to a default.
    """

    def __init__(self, node_answers: dict[str, str], default: str = "<answer>B</answer>"):
        self.node_answers = node_answers
        self.default = default
        self._current_node = ""
        self.prompts: list[str] = []
        #: (node id, prompt) for every call, shared by every per-node copy.
        self.log: list[tuple[str, str]] = []
        self.calls: int = 0

    def for_node(self, node_id: str, role: str = "") -> "_NodeScriptedModel":
        copy = _NodeScriptedModel(self.node_answers, self.default)
        copy._current_node = node_id
        copy.prompts = self.prompts
        copy.log = self.log
        # Share call counter
        copy._counter = self
        return copy

    def with_tool_policy(self, policy: str) -> "_NodeScriptedModel":
        return self

    def with_wall(self, wall: Any) -> "_NodeScriptedModel":
        return self

    def with_budget(self, budget: Any) -> "_NodeScriptedModel":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", workdir: Any = None,
                 cwd: Any = None, env: Any = None, confine: bool = False) -> str:
        self.prompts.append(prompt)
        self.log.append((self._current_node, prompt))
        counter = getattr(self, "_counter", self)
        counter.calls = getattr(counter, "calls", 0) + 1
        return self.node_answers.get(self._current_node, self.default)


class _SequentialModel:
    """A model that returns answers sequentially, like FakeModel."""

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def for_node(self, node_id: str, role: str = "") -> "_SequentialModel":
        return self

    def with_tool_policy(self, policy: str) -> "_SequentialModel":
        return self

    def with_wall(self, wall: Any) -> "_SequentialModel":
        return self

    def with_budget(self, budget: Any) -> "_SequentialModel":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", workdir: Any = None,
                 cwd: Any = None, env: Any = None, confine: bool = False) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.answers.pop(0) if self.answers else "<answer>X</answer>"


class _FailOnceModel:
    """A model that raises on the first call, then answers normally."""

    def __init__(self, first_error: Exception, then: str = "<answer>B</answer>"):
        self._first_error = first_error
        self._then = then
        self._calls = 0

    def for_node(self, node_id: str, role: str = "") -> "_FailOnceModel":
        return self

    def with_tool_policy(self, policy: str) -> "_FailOnceModel":
        return self

    def with_wall(self, wall: Any) -> "_FailOnceModel":
        return self

    def with_budget(self, budget: Any) -> "_FailOnceModel":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", workdir: Any = None,
                 cwd: Any = None, env: Any = None, confine: bool = False) -> str:
        self._calls += 1
        if self._calls == 1:
            raise self._first_error
        return self._then


def _run_walker(mode, model, tmp_path, monkeypatch, *, parallelism: int = 1):
    """Run a mode on the walker with fixture data."""
    _patch_task(monkeypatch)
    if parallelism != 1:
        from refactory_dsh.nn import edits
        state = mode.state_dict()
        state = edits.configure(state, "data_parallelism", str(parallelism))
        mode.load_state_dict(state)
    workdir = tmp_path / "run"
    workdir.mkdir(parents=True, exist_ok=True)
    result = mode.run(workdir, model=model)
    return result


def _spine(graph, workdir, tmp_path, *, agent_responses=None, mode_state=None,
           transform=None):
    """Lower and run graph through the real spine drain."""
    workdir.mkdir(parents=True, exist_ok=True)
    if mode_state is not None:
        state = mode_state
    else:
        state = {"graph": graph.to_state()}
    if not (workdir / ".refactory" / "declarations.json").exists():
        lowering.write_declarations(state, workdir)
    graph_path, _ = lowering.write_mode_graph(
        state, tmp_path / f"graphs-{workdir.name}", name="livemath-blocks",
        **({"transform": transform} if transform is not None else {}))
    run_env = {
        **os.environ,
        "REFACTORY_WORKDIR": str(workdir),
        "PYTHONPATH": str(Path(__file__).resolve().parent),
        # The spine subprocess needs to find the LiveMath fixture data
        "LIVEMATH_DATA_PATH": str(FIXTURE_DIR),
    }
    run_env["PATH"] = f"{PYTHON_BIN}:{run_env.get('PATH', '')}"
    argv = [str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT),
            str(graph_path), str(workdir), f"{PYTHON_BIN}/python", "60000"]
    if agent_responses is not None:
        responses = tmp_path / f"responses-{workdir.name}.json"
        responses.write_text(json.dumps(agent_responses))
        argv.append(str(responses))
    done = subprocess.run(argv, capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=180, env=run_env)
    if done.returncode != 0:
        sys.stderr.write(
            f"spine drain: rc={done.returncode}\n"
            f"stdout: {done.stdout[-2000:]}\n"
            f"stderr: {done.stderr[-3000:]}\n"
        )
    return done.returncode


def _facts(workdir):
    return json.loads((workdir / "facts.json").read_text())


def _scores(records):
    return sorted((str(r["id"]), r.get("score")) for r in records)


def _make_route_client():
    """A real DshModelClient with only the fields routing reads."""
    from refactory_dsh.model import DshModelClient

    client = DshModelClient.__new__(DshModelClient)
    client._worker_backend, client._provider, client._model = "dsh", "anthropic", "claude-opus-4-6"
    client._effort, client._permission_mode, client._max_completion_tokens = "medium", "skip", ""
    client._role_routes, client._node_routes = {}, {}
    client._node_id, client._role, client._dsh_home = "", "worker", Path(".")
    return client


# ══════════════════════════════════════════════════════════════════════════
# 1. Block construction and graph validation
# ══════════════════════════════════════════════════════════════════════════


#: sha256 of tests/lowering_snapshots/livemath-solver.graph.json on main before the
#: block library: the baseline must lower to exactly these bytes.
BASELINE_SNAPSHOT_SHA256 = "c148d36b76daf979b3129679c3bf0e8a3a85b0a10c8a177ecb1e1037fed0ce13"


@pytest.mark.parametrize("explicit", [False, True])
def test_baseline_block_lowers_to_the_frozen_snapshot(explicit):
    """The baseline (built through blocks.build_baseline) lowers byte for byte to the
    committed livemath-solver snapshot, and that snapshot is main's. Proves PIT-003."""
    import hashlib

    from test_lowering_smoke import SNAPSHOT_DIR, _lower_mode_bytes
    from refactory_packages.livemath.mode import Solver

    snapshot = (SNAPSHOT_DIR / "livemath-solver.graph.json").read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == BASELINE_SNAPSHOT_SHA256
    mode_cls = (lambda: Solver(block="baseline")) if explicit else Solver
    assert _lower_mode_bytes(mode_cls, name="livemath-solver") == snapshot


def test_each_block_builds_valid_graph():
    """Every block name produces a valid graph with data, format_prompt, and finish."""
    from refactory_packages.livemath.blocks import BLOCK_NAMES
    from refactory_packages.livemath.mode import Solver

    for name in BLOCK_NAMES:
        mode = Solver(block=name)
        graph = mode.build()
        data_node = graph.nodes["data"]
        assert data_node.subgraph_entry in graph.nodes, f"{name}: entry not in graph"
        assert data_node.subgraph_exit in graph.nodes, f"{name}: exit not in graph"
        assert "finish" in graph.nodes, f"{name}: no finish node"


def test_each_block_exit_writes_answer_txt():
    """Every block's exit node writes answer.txt. Proves PIT-001."""
    from refactory_packages.livemath.blocks import BLOCK_NAMES
    from refactory_packages.livemath.mode import Solver

    for name in BLOCK_NAMES:
        mode = Solver(block=name)
        graph = mode.build()
        exit_id = graph.nodes["data"].subgraph_exit
        exit_node = graph.nodes[exit_id]
        assert "answer.txt" in exit_node.writes, (
            f"{name}: exit node {exit_id!r} does not write answer.txt"
        )


def test_fork_block_has_disjoint_branch_writes():
    """The two_attempts_then_judge block has disjoint writes and produces. Proves PIT-002."""
    from refactory_packages.livemath.mode import Solver

    graph = Solver(block="two_attempts_then_judge").build()
    a = graph.nodes["solve_a"]
    b = graph.nodes["solve_b"]
    assert set(a.writes) & set(b.writes) == set(), "overlapping writes"
    assert set(a.produces) & set(b.produces) == set(), "overlapping produces"
    assert "attempt_a.txt" in a.writes
    assert "attempt_b.txt" in b.writes


def test_unknown_block_rejected():
    """An unknown block name raises ValueError, not silently defaults."""
    from refactory_packages.livemath.mode import Solver

    with pytest.raises(ValueError, match="unknown block"):
        Solver(block="nonexistent")


# ══════════════════════════════════════════════════════════════════════════
# 2. Baseline prompt parity with SkillOpt
# ══════════════════════════════════════════════════════════════════════════


def test_baseline_prompts_match_skillopt():
    """The baseline block's solver prompts equal SkillOpt's _build_system/_build_user
    output for a sample item."""
    import types

    from refactory_dsh.nn.executor import render
    from refactory_packages.livemath.mode import SYSTEM_PROMPT, Solver
    from refactory_packages.livemath.ops import build_user_prompt, skill_section

    # SkillOpt's rollout_system.md, byte for byte
    skillopt_system_template = (
        "You are an expert mathematical reasoning agent solving multiple-choice questions.\n"
        "\n"
        "{skill_section}## Task Format\n"
        "You will receive one mathematics multiple-choice question and its answer choices.\n"
        "Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality"
        " conditions.\n"
        "\n"
        "## Answer Format\n"
        "Think step by step, then provide your final answer inside <answer>...</answer> tags.\n"
        "Inside the tags, output only the single choice label, such as A or C.\n"
        "\n"
        "Example:\n"
        "<answer>B</answer>\n"
    )

    # Check template equivalence
    assert SYSTEM_PROMPT == skillopt_system_template.replace(
        "{skill_section}", "{{skill_section}}"
    )

    # Check rendered output for a sample skill
    skill = "Check edge cases."
    ss = skill_section(skill)
    expected_system = skillopt_system_template.format(skill_section=ss)
    node = Solver().build().nodes["solve"]
    ctx = types.SimpleNamespace(knobs={"skill": skill}, facts={"skill_section": ss, "user_prompt": ""})
    rendered = render(node.system_prompt_template, ctx)
    assert rendered == expected_system


# ══════════════════════════════════════════════════════════════════════════
# 3. Every agent node gets the same route settings
# ══════════════════════════════════════════════════════════════════════════


def test_all_agent_nodes_have_route_settings():
    """Every agent node in every block has tool_policy='none', timeout_s=SOLVE_TIMEOUT_S,
    and SYSTEM_PROMPT with {{skill_section}}."""
    from refactory_packages.livemath.blocks import BLOCK_NAMES
    from refactory_packages.livemath.mode import SOLVE_TIMEOUT_S, SYSTEM_PROMPT, Solver

    for name in BLOCK_NAMES:
        mode = Solver(block=name)
        graph = mode.build()
        for node_id, node in graph.nodes.items():
            if node.kind != "agent":
                continue
            assert node.tool_policy == "none", (
                f"{name}/{node_id}: tool_policy is {node.tool_policy!r}")
            assert node.timeout_s == SOLVE_TIMEOUT_S, (
                f"{name}/{node_id}: timeout_s is {node.timeout_s}")
            assert node.system_prompt_template == SYSTEM_PROMPT, (
                f"{name}/{node_id}: system_prompt_template differs")
            assert "{{skill_section}}" in node.system_prompt_template, (
                f"{name}/{node_id}: missing {{{{skill_section}}}} in system prompt")


@pytest.mark.parametrize("block_name", [
    "baseline", "solve_then_verify", "two_attempts_then_judge", "solve_then_self_check",
])
def test_every_agent_node_uses_solve_timeout_s(block_name):
    """A subclass's solve_timeout_s reaches every agent node of every block."""
    from refactory_packages.livemath.mode import Solver

    class Short(Solver):
        solve_timeout_s = 60

    graph = Short(block=block_name).build()
    agents = [n for n in graph.nodes.values() if n.kind == "agent"]
    assert agents and all(n.timeout_s == 60 for n in agents), (
        [(n.id, n.timeout_s) for n in agents])


def test_max_completion_tokens_applies_to_all_agent_nodes(monkeypatch):
    """The max_completion_tokens knob is applied to every agent node via Solver.run()."""
    from refactory_packages.livemath.blocks import BLOCK_NAMES, BLOCKS
    from refactory_packages.livemath.mode import Solver

    for name in BLOCK_NAMES:
        mode = Solver(block=name)
        from refactory_dsh.nn import edits
        state = mode.state_dict()
        state = edits.configure(state, "max_completion_tokens", "8192")
        mode.load_state_dict(state)

        # Capture the model run() hands to super
        captured = []
        original_run = nn.Mode.run

        def capture_run(self, workdir, *, model=None, **kw):
            captured.append(model)
            return nn.RunResult(exit_ok=True, facts={}, workdir=workdir)

        monkeypatch.setattr(nn.Mode, "run", capture_run)
        try:
            client = _make_route_client()
            mode.run(Path("/tmp/unused"), model=client)
        finally:
            monkeypatch.setattr(nn.Mode, "run", original_run)

        assert captured, f"{name}: no model captured"
        result_model = captured[0]
        routes = getattr(result_model, "_node_routes", {})
        expected_ids = [n.id for n in mode.graph().nodes.values() if n.kind == "agent"]
        assert sorted(expected_ids) == sorted(BLOCKS[name]().agent_node_ids)
        for node_id in expected_ids:
            assert node_id in routes, f"{name}: {node_id} not in routes"
            assert routes[node_id].get("max_completion_tokens") == "8192", (
                f"{name}: {node_id} cap not set")
        captured.clear()


@pytest.mark.parametrize("make", ["Solver", "SolverVerify"])
def test_max_completion_tokens_caps_every_agent_of_a_clone(make, monkeypatch):
    """A clone (``type(self)(state=...)``, as ``Mode.clone`` and the loop make it) of a
    verify block caps solve_draft and verify, though it never saw the block name."""
    from refactory_dsh.nn import edits
    from refactory_packages.livemath import mode as mode_mod

    mode = (mode_mod.Solver(block="solve_then_verify") if make == "Solver"
            else mode_mod.SolverVerify())
    mode.load_state_dict(edits.configure(mode.state_dict(), "max_completion_tokens", "512"))
    clone = type(mode)(state=mode.state_dict())
    captured = []
    monkeypatch.setattr(nn.Mode, "run", lambda self, workdir, *, model=None, **kw: (
        captured.append(model) or nn.RunResult(exit_ok=True, facts={}, workdir=workdir)))
    clone.run(Path("/tmp/unused"), model=_make_route_client())
    routes = captured[0]._node_routes
    for node_id in ("solve_draft", "verify"):
        assert routes[node_id]["max_completion_tokens"] == "512", (node_id, routes)


# ══════════════════════════════════════════════════════════════════════════
# 4. Walker integration for each block
# ══════════════════════════════════════════════════════════════════════════


def test_baseline_walker_reproduces_existing_records(tmp_path, monkeypatch):
    """The baseline block produces the same item_scores as the existing walker fixture."""
    from tests_support import FakeModel
    from refactory_packages.livemath.mode import Solver

    mode = Solver(block="baseline")
    model = FakeModel(["<answer>B</answer>"] * 10)
    result = _run_walker(mode, model, tmp_path, monkeypatch)
    assert result.exit_ok, f"run failed: {result.error}"
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 5

    # Items with correct label B score 1.0: 202501:1, 202503:1, 202503:2
    correct_b_ids = {"202501:1", "202503:1", "202503:2"}
    for entry in scores:
        if entry["id"] in correct_b_ids:
            assert entry["score"] == 1.0, f"{entry['id']} should score 1.0"
        else:
            assert entry["score"] == 0.0, f"{entry['id']} should score 0.0"


def test_solve_then_verify_walker(tmp_path, monkeypatch):
    """The verifier's answer becomes the final answer.txt."""
    from refactory_packages.livemath.mode import Solver

    # Draft returns B, verifier overrides to C
    answers = []
    for _ in range(5):
        answers.append("Draft thinking...\n<answer>B</answer>")  # solve_draft
        answers.append("<answer>C</answer>")  # verify
    model = _SequentialModel(answers)
    mode = Solver(block="solve_then_verify")
    result = _run_walker(mode, model, tmp_path, monkeypatch)
    assert result.exit_ok, f"run failed: {result.error}"

    # All items should have prediction C (from verifier)
    for rec in result.item_records:
        assert rec["prediction"] == "C", f"{rec['id']}: prediction is {rec['prediction']}"


def test_two_attempts_then_judge_walker(tmp_path, monkeypatch):
    """The judge's answer becomes the final answer.txt."""
    from refactory_packages.livemath.mode import Solver

    # 3 calls per item: solve_a, solve_b, judge
    answers = []
    for _ in range(5):
        answers.append("<answer>A</answer>")  # solve_a
        answers.append("<answer>B</answer>")  # solve_b
        answers.append("<answer>D</answer>")  # judge picks D
    model = _SequentialModel(answers)
    mode = Solver(block="two_attempts_then_judge")
    result = _run_walker(mode, model, tmp_path, monkeypatch)
    assert result.exit_ok, f"run failed: {result.error}"

    # All items should have prediction D (from judge)
    for rec in result.item_records:
        assert rec["prediction"] == "D", f"{rec['id']}: prediction is {rec['prediction']}"


def test_self_check_walker(tmp_path, monkeypatch):
    """The self-check sees the SkillOpt refinement text and its answer is final."""
    from refactory_packages.livemath.mode import Solver

    # 2 calls per item: solve_draft, self_check
    answers = []
    for _ in range(5):
        answers.append("<answer>A</answer>")  # solve_draft
        answers.append("<answer>C</answer>")  # self_check overrides to C
    model = _SequentialModel(answers)
    mode = Solver(block="solve_then_self_check")
    result = _run_walker(mode, model, tmp_path, monkeypatch)
    assert result.exit_ok, f"run failed: {result.error}"

    # The self-check prompt should contain the SkillOpt refinement text
    assert any("Re-evaluate the exact option wording. If needed, correct it." in p
               for p in model.prompts), "self-check prompt does not contain refinement text"

    for rec in result.item_records:
        assert rec["prediction"] == "C", f"{rec['id']}: prediction is {rec['prediction']}"


def test_self_check_message_matches_skillopt():
    """The self-check message is SkillOpt's exec-path refinement turn
    (rollout.py:88-95: the full question, ``## Previous Attempt``, the previous
    response, the instruction) plus its answer-format line (rollout.py:102)."""
    from refactory_packages.livemath.block_ops import self_check_message

    assert self_check_message("## Question\nQ\n\n## Choices\nA. x", "prev") == (
        "## Question\nQ\n\n## Choices\nA. x\n\n## Previous Attempt\nprev\n\n"
        "Re-evaluate the exact option wording. If needed, correct it.\n\n"
        "Output only the final choice label inside <answer>...</answer>."
    )


# ── every agent call is framed as arm A's backend frames it ──────────────

_SOLVE_NODES = ("solve", "solve_draft", "solve_a", "solve_b")


def _prompts_by_node(block_name, tmp_path, monkeypatch, node_answers=None):
    """Run a block on the walker; return {node: [(item question, prompt), ...]}."""
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.ops import unwrap_claude_backend_prompt

    model = _NodeScriptedModel(node_answers or {})
    result = _run_walker(Solver(block=block_name), model, tmp_path / block_name, monkeypatch)
    assert result.exit_ok, result.error
    by_node: dict[str, list[str]] = {}
    for node, prompt in model.log:
        by_node.setdefault(node, []).append(prompt)
    return by_node, unwrap_claude_backend_prompt


@pytest.mark.parametrize("block_name", [
    "solve_then_verify", "two_attempts_then_judge", "solve_then_self_check",
])
def test_every_agent_prompt_is_framed_and_solves_match_baseline(block_name, tmp_path,
                                                                 monkeypatch):
    """Every agent call's prompt starts ``History:\n- User: `` and ends
    ``\n\nAnswer the latest user request.``; the solving calls send byte for byte
    the prompts the baseline's solve sends."""
    baseline, _ = _prompts_by_node("baseline", tmp_path, monkeypatch)
    by_node, _ = _prompts_by_node(block_name, tmp_path, monkeypatch)
    assert set(baseline) == {"solve"}
    for node, prompts in by_node.items():
        for prompt in prompts:
            assert prompt.startswith("History:\n- User: "), (node, prompt[:60])
            assert prompt.endswith("\n\nAnswer the latest user request."), (node, prompt[-60:])
    solves = [n for n in by_node if n in _SOLVE_NODES]
    assert solves and len(solves) < len(by_node), by_node.keys()
    for node in solves:
        assert sorted(by_node[node]) == sorted(baseline["solve"]), node


def test_self_check_prompt_has_question_choices_and_draft(tmp_path, monkeypatch):
    """The self-check sees the item's question, every choice and the draft."""

    draft = "DRAFT-REASONING-7f3 <answer>A</answer>"
    by_node, unwrap = _prompts_by_node("solve_then_self_check", tmp_path, monkeypatch,
                                       {"solve_draft": draft})
    from refactory_packages.livemath.tasks import load_items

    items = load_items(str(FIXTURE_DIR))
    assert len(by_node["self_check"]) == len(items) == 5
    messages = [unwrap(p) for p in by_node["self_check"]]
    for data in items:
        question = data["question"]
        matching = [m for m in messages if question in m]
        assert len(matching) == 1, question[:60]
        message = matching[0]
        for choice in data["choices"]:
            assert f"{choice['label']}. {choice['text']}" in message
        assert f"## Previous Attempt\n{draft}\n\nRe-evaluate the exact option wording." in message


def test_verify_prompt_has_question_and_draft(tmp_path, monkeypatch):
    draft = "DRAFT-REASONING-9c1 <answer>A</answer>"
    by_node, unwrap = _prompts_by_node("solve_then_verify", tmp_path, monkeypatch,
                                       {"solve_draft": draft})
    for prompt in by_node["verify"]:
        message = unwrap(prompt)
        assert "## Question\n" in message and "## Choices\n" in message
        assert f"## Solver's Answer\n{draft}" in message


def test_judge_prompt_has_both_distinct_attempts(tmp_path, monkeypatch):
    """The judge sees the question, attempt A's text under ``## Attempt A`` and
    attempt B's under ``## Attempt B``."""
    a, b = "ATTEMPT-ONE-41d <answer>A</answer>", "ATTEMPT-TWO-8e2 <answer>C</answer>"
    by_node, unwrap = _prompts_by_node("two_attempts_then_judge", tmp_path, monkeypatch,
                                       {"solve_a": a, "solve_b": b})
    assert len(by_node["judge"]) == 5
    for prompt in by_node["judge"]:
        message = unwrap(prompt)
        assert "## Question\n" in message
        assert f"## Attempt A\n{a}\n\n## Attempt B\n{b}\n\n" in message


# ── walker with parallelism > 1 ──────────────────────────────────────────


@pytest.mark.parametrize("block_name", [
    "baseline", "solve_then_verify", "two_attempts_then_judge", "solve_then_self_check",
])
def test_blocks_at_parallelism_3_walker(block_name, tmp_path, monkeypatch):
    """Each block runs at parallelism 3 on the walker."""
    from refactory_packages.livemath.mode import Solver

    mode = Solver(block=block_name)
    # Use node-scripted model for consistent answers regardless of order
    node_answers = {
        "solve": "<answer>B</answer>",
        "solve_draft": "<answer>B</answer>",
        "verify": "<answer>B</answer>",
        "self_check": "<answer>B</answer>",
        "solve_a": "<answer>B</answer>",
        "solve_b": "<answer>B</answer>",
        "judge": "<answer>B</answer>",
    }
    model = _NodeScriptedModel(node_answers)
    result = _run_walker(mode, model, tmp_path, monkeypatch, parallelism=3)
    assert result.exit_ok, f"{block_name} at p=3 failed: {result.error}"
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 5


# ══════════════════════════════════════════════════════════════════════════
# 5. Spine + walker equivalence for each block
# ══════════════════════════════════════════════════════════════════════════


def _walker_and_spine_same(block_name, tmp_path, monkeypatch, *, parallelism=1):
    """Helper: run block on walker and spine, assert same item_records/item_scores."""
    from refactory_packages.livemath.mode import Solver

    # Solving calls answer A, the exit node B: only grading the exit node's answer
    # gives the label-B items (202501:1, 202503:1, 202503:2) a 1.
    node_answers = {
        "solve": "<answer>B</answer>",
        "solve_draft": "Draft: <answer>A</answer>",
        "verify": "<answer>B</answer>",
        "self_check": "<answer>B</answer>",
        "solve_a": "<answer>A</answer>",
        "solve_b": "<answer>A</answer>",
        "judge": "<answer>B</answer>",
    }
    label_b = {"202501:1", "202503:1", "202503:2"}

    # Walker run
    model = _NodeScriptedModel(node_answers)
    mode = Solver(block=block_name)
    _patch_task(monkeypatch)
    if parallelism != 1:
        from refactory_dsh.nn import edits
        state = mode.state_dict()
        state = edits.configure(state, "data_parallelism", str(parallelism))
        mode.load_state_dict(state)
    walker_dir = tmp_path / "walker"
    walker_dir.mkdir()
    walker_result = mode.run(walker_dir, model=model)
    assert walker_result.exit_ok, f"walker failed: {walker_result.error}"

    # Spine run
    spine_dir = tmp_path / "spine"
    graph = mode.graph()
    rc = _spine(graph, spine_dir, tmp_path, agent_responses=node_answers,
                mode_state=mode.state_dict())
    assert rc == 0, f"spine failed with rc={rc}"

    spine_facts = _facts(spine_dir)
    walker_scores = _scores(walker_result.facts["item_scores"])
    spine_scores = _scores(spine_facts["item_scores"])
    assert walker_scores == spine_scores, (
        f"walker {walker_scores} != spine {spine_scores}"
    )
    assert len(walker_scores) == 5
    assert walker_scores == sorted((i, 1.0 if i in label_b else 0.0) for i, _ in walker_scores)
    # Every agent call is one recorded exchange on both runtimes (the grid's model calls).
    agents = sum(1 for n in mode.graph().nodes.values() if n.kind == "agent")
    spine_records = json.loads((spine_dir / "item_records.json").read_text())
    for records in (walker_result.item_records, spine_records):
        assert sorted(len(r.get("exchanges") or ()) for r in records) == [agents] * 5


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_baseline_spine_matches_walker(parallelism, tmp_path, monkeypatch):
    _walker_and_spine_same("baseline", tmp_path, monkeypatch, parallelism=parallelism)


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_verify_block_spine_matches_walker(parallelism, tmp_path, monkeypatch):
    _walker_and_spine_same("solve_then_verify", tmp_path, monkeypatch, parallelism=parallelism)


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_judge_block_spine_matches_walker(parallelism, tmp_path, monkeypatch):
    _walker_and_spine_same("two_attempts_then_judge", tmp_path, monkeypatch, parallelism=parallelism)


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_self_check_block_spine_matches_walker(parallelism, tmp_path, monkeypatch):
    _walker_and_spine_same("solve_then_self_check", tmp_path, monkeypatch, parallelism=parallelism)


# ══════════════════════════════════════════════════════════════════════════
# 6. Failed branch / failed solve → item scores 0
# ══════════════════════════════════════════════════════════════════════════


def test_failed_solve_scores_item_zero(tmp_path, monkeypatch):
    """A failed model call in solve_then_verify scores the item 0."""
    from refactory_packages.livemath.mode import Solver

    model = _FailOnceModel(RuntimeError("model exploded"))
    mode = Solver(block="solve_then_verify")
    result = _run_walker(mode, model, tmp_path, monkeypatch)
    assert result.exit_ok, f"run did not complete: {result.error}"
    scores = result.facts.get("item_scores", [])
    # The first item should score 0 (the error), remaining items should succeed
    first_item = scores[0]
    assert first_item["score"] == 0.0, f"failed item scored {first_item['score']}"
    assert any(s["score"] > 0 for s in scores[1:]), "all other items also failed"


class _NodeFailModel:
    """A model that raises on a specific node for the first item, then answers normally.

    Used to test failed fork branches in two_attempts_then_judge.
    """

    def __init__(self, fail_node: str, fail_error: Exception,
                 default: str = "<answer>B</answer>"):
        self._fail_node = fail_node
        self._fail_error = fail_error
        self._default = default
        self._current_node = ""
        self._node_calls: dict[str, int] = {}

    def for_node(self, node_id: str, role: str = "") -> "_NodeFailModel":
        copy = _NodeFailModel(self._fail_node, self._fail_error, self._default)
        copy._current_node = node_id
        copy._node_calls = self._node_calls
        return copy

    def with_tool_policy(self, policy: str) -> "_NodeFailModel":
        return self

    def with_wall(self, wall: Any) -> "_NodeFailModel":
        return self

    def with_budget(self, budget: Any) -> "_NodeFailModel":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", workdir: Any = None,
                 cwd: Any = None, env: Any = None, confine: bool = False) -> str:
        node = self._current_node
        self._node_calls[node] = self._node_calls.get(node, 0) + 1
        # Fail on the specified node's first call only
        if node == self._fail_node and self._node_calls[node] == 1:
            raise self._fail_error
        return self._default


def test_failed_fork_branch_scores_item_zero_run_continues(tmp_path, monkeypatch):
    """A failed fork branch in two_attempts_then_judge scores the item 0, run continues.

    When one fork branch (solve_a) fails for an item, the item scores 0 per F4,
    and the remaining items continue processing normally.
    """
    from refactory_packages.livemath.mode import Solver

    # solve_a fails on first call → first item scores 0
    model = _NodeFailModel("solve_a", RuntimeError("fork branch exploded"))
    mode = Solver(block="two_attempts_then_judge")
    result = _run_walker(mode, model, tmp_path, monkeypatch)
    assert result.exit_ok, f"run did not complete: {result.error}"
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 5, f"expected 5 scores, got {len(scores)}"
    # The first item should score 0 (the failed branch)
    first_item = scores[0]
    assert first_item["score"] == 0.0, f"failed item scored {first_item['score']}"
    # Remaining items should continue and some should succeed
    assert any(s["score"] > 0 for s in scores[1:]), "all other items also failed"


@pytest.mark.parametrize("parallelism", [1, 3])
def test_failed_fork_branch_at_parallelism(parallelism, tmp_path, monkeypatch):
    """Failed fork branch at both parallelism 1 and 3."""
    from refactory_packages.livemath.mode import Solver

    model = _NodeFailModel("solve_a", RuntimeError("fork branch exploded"))
    mode = Solver(block="two_attempts_then_judge")
    result = _run_walker(mode, model, tmp_path, monkeypatch, parallelism=parallelism)
    assert result.exit_ok, f"run at p={parallelism} did not complete: {result.error}"
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 5
    # First item scores 0
    assert scores[0]["score"] == 0.0


#: A dispatched agent, as the spine runs a worker (``dispatch_command``): writes the
#: node's scripted answer to its outputs in the item's pass folder, except that the
#: first dispatch of the failing node exits 1, as a failed worker does.
_FAKE_WORKER = """
import json, os, sys
from pathlib import Path
node, writes, answer, fail_node, counter = sys.argv[1:6]
if node == fail_node:
    path = Path(counter)
    seen = int(path.read_text()) if path.exists() else 0
    path.write_text(str(seen + 1))
    if seen == 0:
        sys.exit("worker failed")
for name in json.loads(writes):
    target = Path(os.environ["REFACTORY_WORKDIR"]) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(answer)
"""


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_failed_fork_branch_spine(parallelism, tmp_path, monkeypatch):
    """On the spine, one failed solve_a worker scores its item 0 with an error kind,
    and every other item is still judged (the judge's B is graded)."""
    import shlex

    from refactory_packages.livemath.mode import Solver

    mode = Solver(block="two_attempts_then_judge")
    if parallelism != 1:
        from refactory_dsh.nn import edits
        state = mode.state_dict()
        state = edits.configure(state, "data_parallelism", str(parallelism))
        mode.load_state_dict(state)
    worker = tmp_path / "fake_worker.py"
    worker.write_text(_FAKE_WORKER)
    answers = {"solve_a": "<answer>A</answer>", "solve_b": "<answer>A</answer>",
               "judge": "<answer>B</answer>"}

    def dispatch(workflow):
        for node in workflow.nodes.values():
            if node.id in answers:
                node.dispatch_command = " ".join(shlex.quote(part) for part in (
                    f"{PYTHON_BIN}/python", str(worker), node.id, json.dumps(sorted(node.writes)),
                    answers[node.id], "solve_a", str(tmp_path / "solve_a.calls")))
                node.command_timeout_s = 60
        return workflow

    spine_dir = tmp_path / "spine"
    rc = _spine(mode.graph(), spine_dir, tmp_path, agent_responses={},
                mode_state=mode.state_dict(), transform=dispatch)
    assert rc == 0, f"spine failed with rc={rc}"
    scores = dict(_scores(_facts(spine_dir)["item_scores"]))
    records = json.loads((spine_dir / "item_records.json").read_text())
    failed = [r for r in records if r.get("error_kind")]
    assert len(scores) == 5 and len(failed) == 1, records
    assert scores[str(failed[0]["id"])] == 0.0
    label_b = {"202501:1", "202503:1", "202503:2"}
    for item_id, score in scores.items():
        if item_id != str(failed[0]["id"]):
            assert score == (1.0 if item_id in label_b else 0.0), (item_id, score)


# ══════════════════════════════════════════════════════════════════════════
# 7. Grid command output
# ══════════════════════════════════════════════════════════════════════════


def test_grid_shortcut_split():
    """A shortcut item (correct_choice text containing 'stronger result can be proven')
    is classified correctly by the grid's split logic."""
    from refactory_packages.livemath.grid import _shortcut_split

    records = [
        {
            "id": "1", "passed": True, "score": 1.0,
            "metadata": {"correct_choice": {
                "text": "A stronger result can be proven: the sequence converges absolutely",
            }},
        },
        {
            "id": "2", "passed": False, "score": 0.0,
            "metadata": {"correct_choice": {"text": "Diverges"}},
        },
        {
            "id": "3", "passed": True, "score": 1.0,
            "metadata": {"correct_choice": {"text": "Converges"}},
        },
    ]
    shortcut_acc, other_acc, shortcut_count, other_count = _shortcut_split(records)
    assert shortcut_count == 1
    assert other_count == 2
    assert shortcut_acc == 1.0
    assert other_acc == 0.5


def _parts(tmp_path) -> Path:
    from refactory_packages.livemath.tasks import LiveMathTask

    items = LiveMathTask(data_path=str(FIXTURE_DIR), shuffle_choices=False).instances()
    parts_path = tmp_path / "parts.json"
    parts_path.write_text(json.dumps({"val": [item.id for item in items], "test": []}))
    return parts_path


_ALL_ANSWERS = {node: "<answer>B</answer>" for node in (
    "solve", "solve_draft", "verify", "self_check", "solve_a", "solve_b", "judge")}


class _Factory:
    """Stands in for DshModelClient in loop.main: records each route, answers with a stub."""

    def __init__(self, model: Any = None):
        self.routes: list[dict[str, Any]] = []
        self.model = model

    def __call__(self, **kwargs: Any) -> Any:
        self.routes.append(kwargs)
        return self.model or _NodeScriptedModel(_ALL_ANSWERS)


def _grid(tmp_path, monkeypatch, extra=(), factory=None, blocks=None):
    from refactory_packages.livemath.grid import main as grid_main

    _patch_task(monkeypatch)
    factory = factory or _Factory()
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "grid_results.json"
    rc = grid_main([
        "livemath/solver", "--parts", str(_parts(tmp_path)), "--output", str(output),
        "--seed", "42", "--part", "val",
        "--blocks", *(blocks or ["baseline", "solve_then_verify",
                                 "two_attempts_then_judge", "solve_then_self_check"]),
        *extra,
    ], _client_factory=factory)
    return rc, output, factory


def test_grid_command_produces_table(tmp_path, monkeypatch):
    """The grid evaluates every block through loop.main, counts each block's model
    calls (1, 2, 3, 2 per item), and keeps each block's run under <output>.runs/."""
    import refactory_dsh.loop as loop_mod

    calls = []
    real_main = loop_mod.main
    monkeypatch.setattr(loop_mod, "main", lambda argv, **kw: calls.append(argv) or real_main(argv, **kw))
    rc, output, _ = _grid(tmp_path, monkeypatch)
    assert rc == 0
    assert len(calls) == 4, "each block must go through refactory optimize's loop.main"
    result = json.loads(output.read_text())
    assert result["seed"] == 42 and result["part"] == "val" and result["complete"] is True
    rows = {row["block"]: row for row in result["blocks"]}
    expected_calls = {"baseline": 5, "solve_then_verify": 10,
                      "two_attempts_then_judge": 15, "solve_then_self_check": 10}
    for block, calls_expected in expected_calls.items():
        row = rows[block]
        assert row["model_calls"] == calls_expected, (block, row["model_calls"])
        assert len(row["item_records"]) == 5
        assert row["accuracy"] == 0.6  # three label-B items of five
        run_dir = Path(row["run_dir"])
        assert run_dir == tmp_path / "grid_results.runs" / block and run_dir.is_dir()
        for key in ("shortcut_accuracy", "other_accuracy", "errors", "timeouts", "wall_time_s"):
            assert key in row


#: Every route flag the grid forwards. claude_code takes no provider override, so the
#: provider is checked on the DSH route.
_CLAUDE_ARGS = ["--worker-backend", "claude_code", "--model", "claude-opus-4-6",
                "--effort", "medium", "--permission-mode", "dontAsk",
                "--setting-sources", "project",
                "--data-parallelism", "3", "--max-completion-tokens", "4096"]
_DSH_ARGS = ["--worker-backend", "dsh", "--provider", "anthropic", "--model", "claude-opus-4-6",
             "--effort", "high", "--data-parallelism", "2"]
_ROUTE_ARGS = _CLAUDE_ARGS


def _flag_values(args: list[str]) -> dict[str, str]:
    return {args[at][2:].replace("-", "_"): args[at + 1] for at in range(0, len(args), 2)}


@pytest.mark.parametrize("args", [_CLAUDE_ARGS, _DSH_ARGS], ids=["claude_code", "dsh"])
def test_grid_route_flags_reach_every_block(args, tmp_path, monkeypatch):
    """Each route flag reaches every block's model route, and the knob flags its
    run config."""
    rc, output, factory = _grid(tmp_path, monkeypatch, extra=args)
    assert rc == 0
    assert len(factory.routes) == 4
    given = _flag_values(args)
    knob_flags = {"data_parallelism", "max_completion_tokens"}
    for route in factory.routes:
        for flag, value in given.items():
            if flag not in knob_flags:
                assert route[flag] == value, (flag, route[flag])
    for block in ("baseline", "solve_then_verify", "two_attempts_then_judge",
                  "solve_then_self_check"):
        config = json.loads((tmp_path / "grid_results.runs" / block / "config.json").read_text())
        for flag, value in given.items():
            if flag in knob_flags:
                assert config["knobs"][flag] == value, (block, flag)
            else:
                assert config[flag] == value, (block, flag)


def test_grid_through_cli_forwards_skill_route_and_evaluator(tmp_path, monkeypatch):
    """`refactory grid` through cli.main: the skill file's text reaches the block's
    knobs, --setting-sources and --evaluator reach its config, and a failed model call
    shows in the row's errors."""
    from refactory_dsh import cli
    import refactory_dsh.model as model_mod

    _patch_task(monkeypatch)
    factory = _Factory(model=_FailOnceModel(RuntimeError("model exploded")))
    monkeypatch.setattr(model_mod, "DshModelClient", factory)
    import refactory_dsh.loop as loop_mod

    seen: list[list[str]] = []
    real_main = loop_mod.main
    monkeypatch.setattr(loop_mod, "main", lambda argv, **kw: seen.append(argv) or real_main(argv, **kw))
    skill = tmp_path / "skill.txt"
    skill.write_text("Read every option twice.")
    output = tmp_path / "grid_results.json"
    rc = cli.main([
        "grid", "livemath/solver", "--parts", str(_parts(tmp_path)), "--output", str(output),
        "--blocks", "baseline", "--skill", str(skill),
        "--worker-backend", "claude_code", "--model", "claude-opus-4-6",
        "--setting-sources", "project", "--evaluator", "walker",
    ])
    assert rc == 0
    assert factory.routes and factory.routes[0]["setting_sources"] == "project"
    config = json.loads((tmp_path / "grid_results.runs" / "baseline" / "config.json").read_text())
    assert config["knobs"]["skill"] == "Read every option twice."
    assert config["setting_sources"] == "project"
    assert config["evaluator"] == "walker"
    # walker is also the default, so check the flag itself reached optimize's argv.
    assert len(seen) == 1 and "--evaluator" in seen[0], seen
    assert seen[0][seen[0].index("--evaluator") + 1] == "walker"
    row = json.loads(output.read_text())["blocks"][0]
    assert row["errors"] == 1, row["errors"]


@pytest.mark.parametrize("extra", [[], _CLAUDE_ARGS, _DSH_ARGS],
                         ids=["no-flags", "claude_code", "dsh"])
def test_grid_route_matches_refactory_optimize(extra, tmp_path, monkeypatch):
    """A block's resolved route is the one `refactory optimize` resolves from the same
    flags -- with no flags, the route optimize would pick."""
    from refactory_dsh import loop
    from refactory_dsh import strategies
    from refactory_packages.livemath.grid import grid_strategy

    rc, _, grid_factory = _grid(tmp_path / "g", monkeypatch, extra=extra, blocks=["baseline"])
    assert rc == 0
    knobs = []
    route_extra = list(extra)
    for flag in ("--data-parallelism", "--max-completion-tokens"):
        if flag in route_extra:
            at = route_extra.index(flag)
            knobs += ["--knob", f"{flag[2:].replace('-', '_')}={route_extra[at + 1]}"]
            del route_extra[at:at + 2]
    optimize_factory = _Factory()
    (tmp_path / "o").mkdir()
    before = strategies.names()
    with grid_strategy("val", 42) as name:
        rc = loop.main([
            "livemath/solver", "--strategy", name,
            "--runs-root", str(tmp_path / "o" / "loop"), "--parts", str(_parts(tmp_path / "o")),
            "--holdout", "0", "--epochs", "1", "--max-evaluations", "1", "--seed", "42",
            *route_extra, *knobs,
        ], _client_factory=optimize_factory)
    # The grid (above) and the context manager leave the registry as they found it.
    assert strategies.names() == before
    assert "_grid_eval_val_42" not in before
    # An evaluate-only strategy has no trained winner, so optimize exits 1 after
    # evaluating; the grid reads the evaluation either way.
    assert rc in (0, 1)
    assert len(grid_factory.routes) == 1
    assert grid_factory.routes == optimize_factory.routes


class _InterruptingModel(_NodeScriptedModel):
    """Raises KeyboardInterrupt (a Ctrl-C) on the nth call overall."""

    def __init__(self, at: int):
        super().__init__(_ALL_ANSWERS)
        self.at = at
        self.count = [0]
        #: The results file, read at the moment of the interrupt.
        self.output: Path | None = None
        self.on_disk: Any = None

    def for_node(self, node_id: str, role: str = "") -> "_InterruptingModel":
        copy = _InterruptingModel(self.at)
        copy._current_node, copy.log, copy.count = node_id, self.log, self.count
        copy.output, copy._parent = self.output, self
        return copy

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.count[0] += 1
        if self.count[0] == self.at:
            if self.output is not None:
                getattr(self, "_parent", self).on_disk = (
                    json.loads(self.output.read_text()) if self.output.exists() else None)
            raise KeyboardInterrupt
        return super().complete(prompt, **kwargs)


def test_interrupted_grid_keeps_finished_blocks_and_stops(tmp_path, monkeypatch):
    """A Ctrl-C in the second block: the first block's row is already on disk, the
    grid exits 130 without starting another call or block, and the runs stay."""
    model = _InterruptingModel(at=5 + 3)  # baseline's 5 calls, then 3 into verify
    model.output = tmp_path / "grid_results.json"
    rc, output, factory = _grid(tmp_path, monkeypatch, factory=_Factory(model))
    assert rc == 130
    # Written when the baseline finished, before the second block's calls.
    assert model.on_disk is not None, "the first block's row was not on disk yet"
    assert [row["block"] for row in model.on_disk["blocks"]] == ["baseline"]
    result = json.loads(output.read_text())
    assert result["complete"] is False
    assert [row["block"] for row in result["blocks"]] == ["baseline"]
    assert model.count[0] == 8, "a call started after the interrupt"
    assert len(factory.routes) == 2, "a block started after the interrupt"
    runs = tmp_path / "grid_results.runs"
    assert (runs / "baseline").is_dir() and (runs / "solve_then_verify").is_dir()


def test_grid_refuses_to_reuse_run_dirs(tmp_path, monkeypatch):
    rc, _, _ = _grid(tmp_path, monkeypatch, blocks=["baseline"])
    assert rc == 0
    rc, _, _ = _grid(tmp_path, monkeypatch, blocks=["baseline"])
    assert rc == 2


# ══════════════════════════════════════════════════════════════════════════
# 8. Mode registration
# ══════════════════════════════════════════════════════════════════════════


def test_modes_registered():
    """All four modes are registered in the livemath package."""
    from refactory_packages.livemath import PACKAGE

    mode_names = {m.name for m in PACKAGE.modes}
    assert mode_names == {"solver", "solver_verify", "solver_judge", "solver_self_check"}


def test_ops_registered():
    """block_ops is registered in the livemath package."""
    from refactory_packages.livemath import PACKAGE

    assert "refactory_packages.livemath.block_ops" in PACKAGE.ops


# ══════════════════════════════════════════════════════════════════════════
# 9. Lowering works for all blocks
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("block_name", [
    "baseline", "solve_then_verify", "two_attempts_then_judge", "solve_then_self_check",
])
def test_lowering_works_for_all_blocks(block_name):
    """Each block's graph can be lowered to a workflow."""
    from refactory_packages.livemath.mode import Solver

    mode = Solver(block=block_name)
    state = mode.state_dict()
    workflow = lowering.to_workflow(state, name=f"livemath-{block_name}")
    assert workflow.start_node
    assert "data" in workflow.nodes
