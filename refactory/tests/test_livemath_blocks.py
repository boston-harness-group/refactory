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
        self.calls: int = 0

    def for_node(self, node_id: str, role: str = "") -> "_NodeScriptedModel":
        copy = _NodeScriptedModel(self.node_answers, self.default)
        copy._current_node = node_id
        copy.prompts = self.prompts
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


def _spine(graph, workdir, tmp_path, *, agent_responses=None, mode_state=None):
    """Lower and run graph through the real spine drain."""
    workdir.mkdir(parents=True, exist_ok=True)
    if mode_state is not None:
        state = mode_state
    else:
        state = {"graph": graph.to_state()}
    if not (workdir / ".refactory" / "declarations.json").exists():
        lowering.write_declarations(state, workdir)
    graph_path, _ = lowering.write_mode_graph(
        state, tmp_path / f"graphs-{workdir.name}", name="livemath-blocks")
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


def test_baseline_block_graph_is_identical():
    """Solver(block='baseline').build().to_state() == Solver().build().to_state().

    Proves PIT-003: the baseline block is byte-identical with today's Solver."""
    from refactory_packages.livemath.mode import Solver

    baseline_explicit = Solver(block="baseline")
    baseline_default = Solver()
    assert baseline_explicit.build().to_state() == baseline_default.build().to_state()


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


def test_max_completion_tokens_applies_to_all_agent_nodes(monkeypatch):
    """The max_completion_tokens knob is applied to every agent node via Solver.run()."""
    from refactory_packages.livemath.blocks import BLOCK_NAMES
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
        expected_ids = mode._agent_node_ids
        for node_id in expected_ids:
            assert node_id in routes, f"{name}: {node_id} not in routes"
            assert routes[node_id].get("max_completion_tokens") == "8192", (
                f"{name}: {node_id} cap not set")
        captured.clear()


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
    assert any("Re-evaluate the exact option wording" in p for p in model.prompts), (
        "self-check prompt does not contain refinement text"
    )

    for rec in result.item_records:
        assert rec["prediction"] == "C", f"{rec['id']}: prediction is {rec['prediction']}"


def test_self_check_prompt_matches_skillopt():
    """The SELF_CHECK_PROMPT contains SkillOpt's exact refinement text."""
    from refactory_packages.livemath.blocks import SELF_CHECK_PROMPT

    assert "Your previous answer was:" in SELF_CHECK_PROMPT
    assert "{{draft_response}}" in SELF_CHECK_PROMPT
    assert "Re-evaluate the exact option wording" in SELF_CHECK_PROMPT
    assert "<answer>...</answer>" in SELF_CHECK_PROMPT


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

    # Different answers per node so the final answer comes from the exit node
    node_answers = {
        "solve": "<answer>B</answer>",
        "solve_draft": "Draft: <answer>A</answer>",
        "verify": "<answer>C</answer>",       # overrides draft
        "self_check": "<answer>D</answer>",    # overrides draft
        "solve_a": "<answer>A</answer>",
        "solve_b": "<answer>B</answer>",
        "judge": "<answer>C</answer>",         # overrides both
    }

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


def test_grid_command_produces_table(tmp_path, monkeypatch):
    """The grid command produces a JSON table with the expected structure."""
    from refactory_packages.livemath.grid import main as grid_main

    _patch_task(monkeypatch)

    # Create a parts manifest
    parts_path = tmp_path / "parts.json"
    # Load fixture items to get their IDs
    from refactory_packages.livemath.tasks import LiveMathTask
    task = LiveMathTask(data_path=str(FIXTURE_DIR), shuffle_choices=False)
    items = task.instances()
    item_ids = [item.id for item in items]
    parts_path.write_text(json.dumps({"val": item_ids, "test": []}))

    output_path = tmp_path / "grid_results.json"

    # Run with just baseline block to keep it fast
    rc = grid_main([
        "livemath/solver",
        "--parts", str(parts_path),
        "--blocks", "baseline",
        "--output", str(output_path),
        "--seed", "42",
        "--part", "val",
    ])
    assert rc == 0, "grid command failed"
    assert output_path.exists(), "output file not created"

    result = json.loads(output_path.read_text())
    assert result["seed"] == 42
    assert result["part"] == "val"
    assert len(result["blocks"]) == 1

    block = result["blocks"][0]
    assert block["block"] == "baseline"
    assert "accuracy" in block
    assert "shortcut_accuracy" in block
    assert "other_accuracy" in block
    assert "item_records" in block
    assert "model_calls" in block
    assert "errors" in block
    assert "timeouts" in block
    assert "wall_time_s" in block


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
