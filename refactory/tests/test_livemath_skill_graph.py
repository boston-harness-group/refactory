"""Tests for livemath/skill_graph: arm A's seed skill as an explicit four-step graph.

Coverage:
- registration, graph shape (fork/join inside the data pass), knobs and what is frozen
- the seed sections are arm A's seed skill, byte for byte
- every agent node is configured as the baseline solve node (route, timeout, cap)
- each step's system prompt holds only its own section; each user prompt holds the
  question, the choices and the earlier steps' notes, framed as arm A's backend frames it
- the walker and the spine run the graph, fork and join, record every agent call,
  and grade decide's answer with LiveMath's grader
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from test_livemath_blocks import (
    FIXTURE_DIR,
    _facts,
    _make_route_client,
    _patch_task,
    _scores,
    _spine,
    needs_spine,
)

ARM_A_PROMPTS = Path(__file__).resolve().parent / "fixtures" / "skillopt_arm_a_optimizer_prompts.json"
LABEL_B = {"202501:1", "202503:1", "202503:2"}
AGENTS = ("compare", "precision", "hypotheses", "decide")


class _StepModel:
    """Answers per node and logs (node, prompt, system) for every call."""

    def __init__(self, answers: dict[str, str], log: list | None = None, node: str = "",
                 fail: tuple[str, list] | None = None):
        self.answers = answers
        self.log = [] if log is None else log
        self.node = node
        #: (node to fail, shared call counter): that node's first call raises.
        self.fail = fail

    def for_node(self, node_id: str, role: str = "") -> "_StepModel":
        return _StepModel(self.answers, self.log, node_id, self.fail)

    def with_tool_policy(self, policy: str) -> "_StepModel":
        return self

    def with_wall(self, wall: Any) -> "_StepModel":
        return self

    def with_budget(self, budget: Any) -> "_StepModel":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.log.append((self.node, prompt, system))
        if self.fail and self.node == self.fail[0]:
            self.fail[1].append(1)
            if len(self.fail[1]) == 1:
                raise RuntimeError("step exploded")
        return self.answers.get(self.node, "<answer>B</answer>")


def _mode(parallelism: int = 1, **knobs: str):
    from refactory_dsh.nn import edits
    from refactory_packages.livemath.skill_graph import SkillGraph

    mode = SkillGraph()
    state = mode.state_dict()
    if parallelism != 1:
        state = edits.configure(state, "data_parallelism", str(parallelism))
    for name, value in knobs.items():
        state = edits.configure(state, name, value)
    mode.load_state_dict(state)
    return mode


def _walk(mode, model, tmp_path, monkeypatch, name="walker"):
    _patch_task(monkeypatch)
    workdir = tmp_path / name
    workdir.mkdir(parents=True, exist_ok=True)
    result = mode.run(workdir, model=model)
    assert result.exit_ok, result.error
    return result


# ── registration, shape, knobs ────────────────────────────────────────────


def test_skill_graph_is_registered_with_its_ops():
    from refactory_packages.livemath import PACKAGE
    from refactory_packages.livemath.skill_graph import SKILL_GRAPH_OPS, SkillGraph

    assert SkillGraph in PACKAGE.modes and SkillGraph.name == "skill_graph"
    assert SKILL_GRAPH_OPS in PACKAGE.ops
    assert nn.validate_state(SkillGraph().state_dict()) == []


def test_graph_is_compare_then_parallel_steps_then_decide_inside_the_data_pass():
    from refactory_packages.livemath.skill_graph import SkillGraph

    graph = SkillGraph().build()
    data = graph.nodes["data"]
    assert (data.subgraph_entry, data.subgraph_exit) == ("prepare", "extract_answer")
    assert sorted(n.id for n in graph.nodes.values() if n.kind == "agent") == sorted(AGENTS)
    assert graph.nodes["step_fork"].targets == ("precision", "hypotheses")
    assert graph.nodes["step_join"].sources == ("precision", "hypotheses")
    assert sorted((e.source, e.target) for e in graph.edges) == sorted([
        ("prepare", "compare"), ("compare", "notes_prompts"), ("notes_prompts", "step_fork"),
        ("step_fork", "step_join"), ("precision", "step_join"), ("hypotheses", "step_join"),
        ("step_join", "decide_prompt"), ("decide_prompt", "decide"),
        ("decide", "extract_answer"), ("data", "finish"),
    ])
    for op in ("prepare", "notes_prompts", "decide_prompt", "extract_answer"):
        assert graph.nodes[op].kind == "op", op
    # decide writes the file LiveMath's task grades; the branches write disjoint notes.
    assert graph.nodes["decide"].writes == ("answer.txt",)
    assert set(graph.nodes["precision"].writes).isdisjoint(graph.nodes["hypotheses"].writes)


def test_knobs_open_sections_frozen_system_prompts_and_no_skill_knob():
    from refactory_packages.livemath.skill_graph import SEED_SECTIONS, SkillGraph, step_system_prompt

    state = SkillGraph().state_dict()
    knobs = state["knobs"]
    assert "skill" not in knobs
    open_knobs = sorted(n for n, e in knobs.items() if not e["spec"]["frozen"])
    assert open_knobs == ["compare_skill", "decide_skill", "hypotheses_skill",
                          "precision_skill", "use_sketch", "use_theorem"]
    frozen = sorted(n for n, e in knobs.items() if e["spec"]["frozen"])
    assert frozen == sorted(["data_parallelism", "max_completion_tokens",
                             *(f"{s}__system_prompt" for s in AGENTS)])
    graph = nn.Graph.from_state(state["graph"])
    for step in AGENTS:
        assert knobs[f"{step}_skill"]["value"] == SEED_SECTIONS[step]
        assert knobs[f"{step}__system_prompt"]["value"] == step_system_prompt(step)
        assert graph.nodes[step].field_bindings == (("system_prompt_template", f"{step}__system_prompt"),)
        assert graph.nodes[step].bound_knob("prompt_template") == ""
    issues = nn.validate_state(state, resolve_calls=False)
    assert not [i for i in issues if i.code in ("V7", "V14")], issues


def test_seed_sections_are_arm_a_seed_skill():
    """Title + the four sections, joined as the seed file joins them, is arm A's seed
    skill as its first optimizer call quotes it (calls.jsonl line 55)."""
    from refactory_packages.livemath.skill_graph import SEED_SECTIONS, STEPS

    message = json.loads(ARM_A_PROMPTS.read_text())[0]["message"]
    seed = message.split("## Current Skill\n", 1)[1].split("\n\n\n## Edits Budget", 1)[0]
    assert seed == "# Live Mathematical MCQ Heuristics\n\n" + "\n\n".join(
        SEED_SECTIONS[s] for s in STEPS)


def test_every_agent_is_configured_as_the_baseline_solve():
    from refactory_packages.livemath.mode import SOLVE_TIMEOUT_S, Solver
    from refactory_packages.livemath.skill_graph import SkillGraph

    solve = Solver().build().nodes["solve"]
    for cls, timeout in ((SkillGraph, SOLVE_TIMEOUT_S),
                         (type("Short", (SkillGraph,), {"solve_timeout_s": 60}), 60)):
        graph = cls().build()
        for step in AGENTS:
            node = graph.nodes[step]
            assert (node.tool_policy, node.role, node.timeout_s) == (
                solve.tool_policy, solve.role, timeout), step
            assert "max_completion_tokens" in node.knobs


def test_max_completion_tokens_caps_every_agent(monkeypatch):
    from refactory_dsh.nn import edits
    from refactory_packages.livemath.skill_graph import SkillGraph

    mode = SkillGraph()
    mode.load_state_dict(edits.configure(mode.state_dict(), "max_completion_tokens", "512"))
    clone = type(mode)(state=mode.state_dict())
    captured = []
    monkeypatch.setattr(nn.Mode, "run", lambda self, workdir, *, model=None, **kw: (
        captured.append(model) or nn.RunResult(exit_ok=True, facts={}, workdir=workdir)))
    clone.run(Path("/tmp/unused"), model=_make_route_client())
    routes = captured[0]._node_routes
    assert sorted(routes) == sorted(AGENTS)
    assert all(routes[s]["max_completion_tokens"] == "512" for s in AGENTS), routes
    # Uncapped by default: the route is left as it was.
    captured.clear()
    SkillGraph().run(Path("/tmp/unused"), model=_make_route_client())
    assert captured[0]._node_routes == {}


# ── prompts ───────────────────────────────────────────────────────────────


def _calls(tmp_path, monkeypatch, answers=None, **knobs):
    from refactory_packages.livemath.ops import unwrap_claude_backend_prompt

    model = _StepModel(answers or {})
    _walk(_mode(**knobs), model, tmp_path, monkeypatch)
    by_node: dict[str, list[tuple[str, str]]] = {}
    for node, prompt, system in model.log:
        assert prompt.startswith("History:\n- User: "), (node, prompt[:60])
        assert prompt.endswith("\n\nAnswer the latest user request."), (node, prompt[-60:])
        by_node.setdefault(node, []).append((unwrap_claude_backend_prompt(prompt), system))
    return by_node


def test_each_step_has_only_its_own_section_and_the_earlier_notes(tmp_path, monkeypatch):
    from refactory_packages.livemath.mode import SYSTEM_PROMPT
    from refactory_packages.livemath.ops import skill_section
    from refactory_packages.livemath.skill_graph import SEED_SECTIONS
    from refactory_packages.livemath.tasks import load_items

    notes = {"compare": "CMP-NOTES-1a7", "precision": "PREC-NOTES-2b8",
             "hypotheses": "HYP-NOTES-3c9"}
    by_node = _calls(tmp_path, monkeypatch, {**notes, "decide": "<answer>B</answer>"})
    items = load_items(str(FIXTURE_DIR))
    assert {n: len(c) for n, c in by_node.items()} == {s: len(items) for s in AGENTS}
    for step in AGENTS:
        for message, system in by_node[step]:
            # LiveMath's solver system prompt with this step's section as its skill.
            assert system == SYSTEM_PROMPT.replace(
                "{{skill_section}}", skill_section(SEED_SECTIONS[step])), step
            # The question and every choice, as the baseline formats them.
            matching = [d for d in items if d["question"] in message]
            assert len(matching) == 1, (step, message[:80])
            for choice in matching[0]["choices"]:
                assert f"{choice['label']}. {choice['text']}" in message
            earlier = {"compare": (), "precision": ("compare",), "hypotheses": ("compare",),
                       "decide": ("compare", "precision", "hypotheses")}[step]
            for other, text in notes.items():
                assert (text in message) == (other in earlier), (step, other)


def test_compare_sends_the_baseline_question_unchanged(tmp_path, monkeypatch):
    """compare's message starts with the baseline solve's whole message for the item."""
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.ops import unwrap_claude_backend_prompt
    from refactory_packages.livemath.skill_graph import COMPARE_TASK

    baseline = _StepModel({})
    _walk(Solver(), baseline, tmp_path, monkeypatch, name="baseline")
    expected = sorted(unwrap_claude_backend_prompt(p) + "\n\n" + COMPARE_TASK
                      for _, p, _ in baseline.log)
    assert sorted(m for m, _ in _calls(tmp_path, monkeypatch)["compare"]) == expected


def test_a_section_knob_reaches_only_its_step(tmp_path, monkeypatch):
    by_node = _calls(tmp_path, monkeypatch, hypotheses_skill="HYP-SKILL-MARKER",
                     use_theorem="true")
    for step in AGENTS:
        for message, system in by_node[step]:
            assert ("## Skill\nHYP-SKILL-MARKER\n\n" in system) == (step == "hypotheses")
    # use_theorem reaches the question every step sees, as in the baseline (two of
    # the five fixture items carry a theorem).
    for step in AGENTS:
        assert sum("## Theorem\n" in m for m, _ in by_node[step]) == 2, step


# ── ops ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reply, label, valid", [
    ("reasoning...\n<answer>C</answer>", "C", True),
    ("<answer>b.</answer>", "B", True),
    ("no tags, last line names D", "NO TAGS, LAST LINE NAMES D", False),
    ("<answer>Z</answer>", "Z", False),
])
def test_extract_answer_uses_the_graders_rules(tmp_path, reply, label, valid):
    from refactory_packages.livemath.grading import parse_choice_label
    from refactory_packages.livemath.skill_graph import extract_answer

    choices = [{"label": x, "text": f"option {x}"} for x in "ABCD"]
    (tmp_path / "answer.txt").write_text(reply)
    ctx = types.SimpleNamespace(
        read_text=lambda name: (tmp_path / name).read_text(),
        read_json=lambda name, default=None: {"choices": choices})
    out = extract_answer(ctx)
    assert out == {"answer_label": label, "answer_valid": valid}
    assert out["answer_label"] == parse_choice_label(reply, choices)
    assert (tmp_path / "answer.txt").read_text() == reply


# ── both runtimes ─────────────────────────────────────────────────────────

#: Every step but decide answers A; only grading decide's B gives the label-B items a 1.
_ANSWERS = {"compare": "compare notes <answer>A</answer>",
            "precision": "precision notes <answer>A</answer>",
            "hypotheses": "hypotheses notes <answer>A</answer>",
            "decide": "<answer>B</answer>"}


def _expected_scores(records):
    return sorted((str(r["id"]), 1.0 if str(r["id"]) in LABEL_B else 0.0) for r in records)


def _check_records(records):
    assert len(records) == 5
    for record in records:
        exchanges = record.get("exchanges") or []
        assert sorted(e["node"] for e in exchanges) == sorted(AGENTS), record["id"]
        by_node = {e["node"]: e for e in exchanges}
        for step in AGENTS:
            assert by_node[step]["prompt"].startswith("History:\n- User: "), step
            assert by_node[step]["response"] == _ANSWERS[step], step
        # The notes flow: each step's prompt carries exactly the earlier steps' replies.
        for step, earlier in (("compare", ()), ("precision", ("compare",)),
                              ("hypotheses", ("compare",)),
                              ("decide", ("compare", "precision", "hypotheses"))):
            for other in ("compare", "precision", "hypotheses"):
                assert (f"{other} notes" in by_node[step]["prompt"]) == (other in earlier), (
                    step, other)
        assert record["prediction"] == "B"


@pytest.mark.parametrize("parallelism", [1, 3])
def test_walker_runs_forks_joins_and_grades_decide(parallelism, tmp_path, monkeypatch):
    model = _StepModel(_ANSWERS)
    result = _walk(_mode(parallelism), model, tmp_path, monkeypatch)
    assert _scores(result.facts["item_scores"]) == _expected_scores(result.facts["item_scores"])
    _check_records(result.item_records)
    # decide ran after both branches every time.
    order = [n for n, _, _ in model.log]
    assert order.count("decide") == 5


def test_a_failed_branch_scores_its_item_zero_and_the_run_continues(tmp_path, monkeypatch):
    model = _StepModel(_ANSWERS, fail=("hypotheses", []))
    result = _walk(_mode(), model, tmp_path, monkeypatch)
    scores = result.facts["item_scores"]
    failed = [r for r in result.item_records if r.get("error_kind")]
    assert len(scores) == 5 and len(failed) == 1
    assert dict(_scores(scores))[str(failed[0]["id"])] == 0.0
    assert sum(s["score"] for s in scores) == len(LABEL_B - {str(failed[0]["id"])})


@needs_spine
@pytest.mark.parametrize("parallelism", [1, 3])
def test_spine_matches_walker(parallelism, tmp_path, monkeypatch):
    mode = _mode(parallelism)
    walker = _walk(mode, _StepModel(_ANSWERS), tmp_path, monkeypatch)
    spine_dir = tmp_path / "spine"
    rc = _spine(mode.graph(), spine_dir, tmp_path, agent_responses=_ANSWERS,
                mode_state=mode.state_dict())
    assert rc == 0
    spine_scores = _scores(_facts(spine_dir)["item_scores"])
    assert spine_scores == _scores(walker.facts["item_scores"]) == _expected_scores(
        walker.facts["item_scores"])
    spine_records = json.loads((spine_dir / "item_records.json").read_text())
    _check_records(spine_records)
