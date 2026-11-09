"""Tests for livemath/skill_graph: arm A's seed skill as an explicit four-step graph.

Coverage:
- registration, graph shape (fork/join inside the data pass), knobs and what is frozen
- the seed sections are arm A's seed skill, byte for byte
- every agent node is configured as the baseline solve node (route, timeout, cap)
- with default knobs every call's system and user prompt is byte-identical to the
  section-knob graph's (expected strings built here from SYSTEM_PROMPT, SEED_SECTIONS
  and the old task texts)
- each step's system prompt holds only its own section; each user prompt holds the
  question, the choices and the earlier steps' notes, framed as arm A's backend frames it
- each step's system-prompt and task knob reaches only that step, on both runtimes
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


def test_knobs_open_system_prompts_and_tasks_and_no_skill_knob():
    """Per step one open system-prompt knob and one open task knob, nothing else open;
    the agents bind the system prompt, and each prompt builder declares exactly the
    task knobs it reads."""
    from refactory_packages.livemath.skill_graph import TASKS, SkillGraph, step_system_prompt

    state = SkillGraph().state_dict()
    knobs = state["knobs"]
    assert "skill" not in knobs
    assert not [n for n in knobs if n.endswith("_skill") or n.endswith("_skill_section")]
    open_knobs = sorted(n for n, e in knobs.items() if not e["spec"]["frozen"])
    assert open_knobs == sorted([*(f"{s}__system_prompt" for s in AGENTS),
                                 *(f"{s}_task" for s in AGENTS)])
    assert all(knobs[n]["spec"]["kind"] == "str" for n in open_knobs)
    frozen = sorted(n for n, e in knobs.items() if e["spec"]["frozen"])
    assert frozen == sorted(["data_parallelism", "max_completion_tokens",
                             "use_sketch", "use_theorem"])
    graph = nn.Graph.from_state(state["graph"])
    for step in AGENTS:
        assert knobs[f"{step}__system_prompt"]["value"] == step_system_prompt(step)
        assert graph.nodes[step].system_prompt_template == step_system_prompt(step)
        assert "{{" not in step_system_prompt(step)
        assert knobs[f"{step}_task"]["value"] == TASKS[step]
        assert graph.nodes[step].field_bindings == (("system_prompt_template", f"{step}__system_prompt"),)
        assert graph.nodes[step].bound_knob("prompt_template") == ""
        assert graph.nodes[step].knobs == ("max_completion_tokens", f"{step}__system_prompt")
    assert graph.nodes["prepare"].knobs == ("compare_task", "use_theorem", "use_sketch")
    assert graph.nodes["notes_prompts"].knobs == ("precision_task", "hypotheses_task")
    assert graph.nodes["decide_prompt"].knobs == ("decide_task",)
    assert graph.nodes["extract_answer"].knobs == ()
    issues = nn.validate_state(state, resolve_calls=False)
    assert not [i for i in issues if i.code in ("V7", "V14")], issues


@pytest.mark.parametrize("name", ["use_theorem", "use_sketch"])
def test_an_optimizer_edit_to_the_theorem_or_sketch_switch_is_refused(name):
    """An item's theorem states the correct option: the optimizer's set_knob on either
    switch is refused, and the state keeps it off. A system-prompt knob still moves."""
    from refactory_dsh.nn import edits
    from refactory_packages.livemath.skill_graph import SkillGraph

    state = SkillGraph().state_dict()
    with pytest.raises(edits.EditError, match="is frozen and cannot be moved"):
        edits.apply({"op": "set_knob", "name": name, "value": True}, state)
    assert state["knobs"][name]["value"] is False
    moved = edits.apply({"op": "set_knob", "name": "compare__system_prompt", "value": "X"}, state)
    assert moved["knobs"]["compare__system_prompt"]["value"] == "X"


#: Method terms of the seed skill's sections: none may appear in the default task texts.
_SEED_TERMS = ("strongest", "quantifier", "there exists", "for every", "if and only if",
               "exactly when", "distractor", "weaken", "overstat", "characteriz",
               "equivalence", "regularity", "existential", "universal", "hypothes",
               "domain", "assumption", "equality", "extremal", "subfamily", "precision",
               "theorem")


def test_no_seed_guidance_in_the_default_step_task_texts():
    """Every step's default task text carries no seed guidance: no seed method term and
    no four-word run of any seed section, so all seed method guidance starts in the
    system-prompt knobs."""
    import re

    from refactory_packages.livemath import skill_graph as sg

    words = lambda text: re.findall(r"[a-z]+", text.lower())
    runs = {tuple(w[i:i + 4]) for section in sg.SEED_SECTIONS.values()
            for w in [words(section)] for i in range(len(w) - 3)}
    for name in ("COMPARE_TASK", "PRECISION_TASK", "HYPOTHESES_TASK", "DECIDE_TASK"):
        text = getattr(sg, name)
        assert not [t for t in _SEED_TERMS if t in text.lower()], name
        w = words(text)
        assert not {tuple(w[i:i + 4]) for i in range(len(w) - 3)} & runs, name


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
    from refactory_packages.livemath.skill_graph import ANSWER_FORMAT, NOTES_FORMAT, SEED_SECTIONS
    from refactory_packages.livemath.tasks import load_items

    notes = {"compare": "CMP-NOTES-1a7", "precision": "PREC-NOTES-2b8",
             "hypotheses": "HYP-NOTES-3c9"}
    by_node = _calls(tmp_path, monkeypatch, {**notes, "decide": "<answer>B</answer>"})
    items = load_items(str(FIXTURE_DIR))
    assert {n: len(c) for n, c in by_node.items()} == {s: len(items) for s in AGENTS}
    for step in AGENTS:
        for message, system in by_node[step]:
            # LiveMath's solver system prompt with this step's section as its skill
            # (the analysis steps' answer format swapped for the notes format).
            expected = SYSTEM_PROMPT.replace(
                "{{skill_section}}", skill_section(SEED_SECTIONS[step]))
            if step != "decide":
                expected = expected.replace(ANSWER_FORMAT, NOTES_FORMAT)
            assert system == expected, step
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


#: The step task texts the section-knob graph hardcoded (origin/main before the task
#: knobs), copied verbatim so a changed default fails the byte-identity test below.
_OLD_TASKS = {
    "compare": ("## Your Task\n"
                "This is the first of several analysis steps. Apply the skill in your "
                "instructions to every option and write your analysis as notes for the next "
                "steps. A final answer is not required."),
    "precision": ("## Your Task\n"
                  "Using the notes above, apply the skill in your instructions to every option and "
                  "write your analysis as notes for the final step. A final answer is not required."),
    "decide": ("## Your Task\n"
               "Weigh the notes above against the question and its options, and choose the "
               "single correct option. Think step by step, then provide your final answer "
               "inside <answer>...</answer> tags, with only the single choice label inside."),
}
_OLD_TASKS["hypotheses"] = _OLD_TASKS["precision"]


def _old_calls(questions: list[str], notes: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Every (framed user prompt, system prompt) the section-knob graph sent per step,
    built independently: SYSTEM_PROMPT with the step's seed section as its ``## Skill``
    block (the analysis steps' answer format swapped for the notes format), and the
    question, the earlier notes under their headings and the old task."""
    from refactory_packages.livemath.mode import SYSTEM_PROMPT
    from refactory_packages.livemath.ops import claude_backend_prompt
    from refactory_packages.livemath.skill_graph import SEED_SECTIONS

    heads = {"compare": "## Option Comparison Notes", "precision": "## Theorem-Level Precision Notes",
             "hypotheses": "## Hypotheses Notes"}
    earlier = {"compare": (), "precision": ("compare",), "hypotheses": ("compare",),
               "decide": ("compare", "precision", "hypotheses")}
    out: dict[str, list[tuple[str, str]]] = {}
    answer_format = SYSTEM_PROMPT[SYSTEM_PROMPT.index("## Answer Format\n"):]
    notes_format = ("## Output Format\n"
                    "Think step by step and write your analysis as notes for the later steps.\n"
                    "Do not choose a final answer and do not write <answer> tags; the final "
                    "step decides.\n")
    for step in AGENTS:
        system = SYSTEM_PROMPT.replace("{{skill_section}}",
                                       "## Skill\n" + SEED_SECTIONS[step].strip() + "\n\n")
        if step != "decide":
            system = system.replace(answer_format, notes_format)
        for question in questions:
            blocks = [question, *(f"{heads[o]}\n{notes[o].strip()}" for o in earlier[step]),
                      _OLD_TASKS[step]]
            out.setdefault(step, []).append((claude_backend_prompt("\n\n".join(blocks)), system))
        out[step].sort()
    return out


def test_default_knobs_send_byte_identical_prompts(tmp_path, monkeypatch):
    """With default knobs every agent call's system prompt and framed user prompt is
    byte-identical to the expected one (decide's system prompt is the solver's own),
    for every fixture item and all four steps. Fails when a system-prompt or task
    default drifts by one byte."""
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.ops import unwrap_claude_backend_prompt

    baseline = _StepModel({})
    _walk(Solver(), baseline, tmp_path, monkeypatch, name="baseline")
    questions = [unwrap_claude_backend_prompt(p) for _, p, _ in baseline.log]
    assert len(questions) == 5
    notes = {"compare": "CMP notes\n", "precision": " PREC notes", "hypotheses": "HYP notes"}
    model = _StepModel({**notes, "decide": "<answer>B</answer>"})
    _walk(_mode(), model, tmp_path, monkeypatch)
    sent: dict[str, list[tuple[str, str]]] = {}
    for node, prompt, system in model.log:
        sent.setdefault(node, []).append((prompt, system))
    assert {k: sorted(v) for k, v in sent.items()} == _old_calls(questions, notes)


def test_only_decide_is_asked_for_an_answer():
    """The analysis steps' default system prompts ask for notes and carry no answer
    format; decide's is the solver's prompt with its seed section.
    Fails when an analysis step is told to answer again."""
    from refactory_packages.livemath.mode import SYSTEM_PROMPT
    from refactory_packages.livemath.skill_graph import NOTES_FORMAT, step_system_prompt

    for step in ("compare", "precision", "hypotheses"):
        prompt = step_system_prompt(step)
        assert "## Answer Format" not in prompt and "<answer>B</answer>" not in prompt, step
        assert "provide your final answer" not in prompt, step
        assert prompt.endswith(NOTES_FORMAT), step
    decide = step_system_prompt("decide")
    assert "## Answer Format" in decide and decide.endswith(SYSTEM_PROMPT.split("{{skill_section}}")[1])


def test_a_system_prompt_or_task_knob_reaches_only_its_step(tmp_path, monkeypatch):
    """A step's system-prompt knob is that step's whole system prompt, and its task
    knob ends that step's user message; no other step sees either."""
    from refactory_packages.livemath.skill_graph import step_system_prompt

    by_node = _calls(tmp_path, monkeypatch, hypotheses__system_prompt="HYP-SYSTEM-MARKER",
                     use_theorem="true", **{f"{s}_task": f"{s.upper()}-TASK-MARKER" for s in AGENTS})
    for step in AGENTS:
        for message, system in by_node[step]:
            assert system == ("HYP-SYSTEM-MARKER" if step == "hypotheses"
                              else step_system_prompt(step)), step
            assert message.endswith(f"\n\n{step.upper()}-TASK-MARKER"), step
            assert message.count("-TASK-MARKER") == 1, step
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


@needs_spine
def test_spine_carries_the_system_prompt_and_task_knobs(tmp_path, monkeypatch):
    """Lowered for the spine, an edited system-prompt knob is that agent's system
    prompt and an edited task knob ends that step's user message, as on the walker;
    the other steps keep their defaults."""
    from refactory_dsh import lowering
    from refactory_packages.livemath.ops import unwrap_claude_backend_prompt
    from refactory_packages.livemath.skill_graph import step_system_prompt

    _patch_task(monkeypatch)
    mode = _mode(compare__system_prompt="CMP-SYSTEM-MARKER", hypotheses_task="HYP-TASK-MARKER")
    # Op commands read the knobs from the environment, as ``dsh_runtime`` passes them.
    monkeypatch.setenv(lowering.KNOBS_ENV,
                       lowering.knob_environment(mode.state_dict())[lowering.KNOBS_ENV])
    spine_dir = tmp_path / "spine-knobs"
    rc = _spine(mode.graph(), spine_dir, tmp_path, agent_responses=_ANSWERS,
                mode_state=mode.state_dict())
    assert rc == 0
    lowered = json.loads(next((tmp_path / f"graphs-{spine_dir.name}").glob("*.graph.json"))
                         .read_text())["nodes"]
    for step in AGENTS:
        assert lowered[step]["system_prompt"] == (
            "CMP-SYSTEM-MARKER" if step == "compare" else step_system_prompt(step)), step
    records = json.loads((spine_dir / "item_records.json").read_text())
    assert len(records) == 5
    for record in records:
        for exchange in record["exchanges"]:
            message = unwrap_claude_backend_prompt(exchange["prompt"])
            is_hyp = exchange["node"] == "hypotheses"
            assert message.endswith("\n\nHYP-TASK-MARKER") == is_hyp, exchange["node"]
            assert ("HYP-TASK-MARKER" in message) == is_hyp, exchange["node"]


# ── a {{fact}} in a step's task knob ───────────────────────────────────────


def _notes_dir(path: Path) -> Path:
    """A work directory holding the three steps' notes, as decide_prompt reads them
    (made-up text: no item is read)."""
    path.mkdir(parents=True)
    for name in ("compare_notes.txt", "precision_notes.txt", "hypotheses_notes.txt"):
        (path / name).write_text(f"{name} body")
    return path


def test_a_fact_in_a_task_knob_reaches_its_step_on_both_runtimes(tmp_path, monkeypatch):
    """A `{{name}}` in a step's task knob is filled from the run's facts, as an agent's
    prompt is: the walker hands the op its facts, the spine's runner reads them from
    facts.json and the knobs from the environment, and both send the filled text."""
    from refactory_dsh import lowering
    from refactory_dsh.run import FACTS, build_op_context, run_node
    from refactory_packages.livemath.ops import unwrap_claude_backend_prompt
    from refactory_packages.livemath.skill_graph import SKILL_GRAPH_OPS, decide_prompt

    state = _mode(decide_task="Weigh {{conclusions}} first.").state_dict()
    knobs = {name: entry["value"] for name, entry in state["knobs"].items()}
    facts = {"user_prompt": "THE QUESTION", "conclusions": "CONCLUSIONS-MARKER"}
    walker_dir = _notes_dir(tmp_path / "walker")
    walked = decide_prompt(build_op_context(walker_dir, "decide_prompt", knobs=knobs,
                                            facts=facts))["decide_prompt"]
    spine_dir = _notes_dir(tmp_path / "spine")
    (spine_dir / FACTS).write_text(json.dumps(facts))
    # What lowering writes for the spine: the node's declarations, keyed by id.
    declared = spine_dir / lowering.DECLARATIONS
    declared.parent.mkdir(parents=True, exist_ok=True)
    node = _mode().graph().nodes["decide_prompt"]
    declared.write_text(json.dumps({"decide_prompt": {
        "reads": list(node.reads), "writes": list(node.writes),
        "produces": list(node.produces)}}))
    monkeypatch.setenv(lowering.KNOBS_ENV, lowering.knob_environment(state)[lowering.KNOBS_ENV])
    run_node(f"{SKILL_GRAPH_OPS}:decide_prompt", workdir=spine_dir, node_id="decide_prompt")
    spun = json.loads((spine_dir / FACTS).read_text())["decide_prompt"]
    assert walked == spun
    message = unwrap_claude_backend_prompt(walked)
    assert message.endswith("\n\nWeigh CONCLUSIONS-MARKER first."), message[-120:]
    assert message.startswith("THE QUESTION") and "{{" not in message


def test_a_task_knob_slot_is_held_to_what_is_produced_before_its_op():
    """The ops that read the task knobs fill facts (`nn.fills_facts`), so a slot there is
    one of the op's inputs: a fact produced before the op is accepted, one that is not is
    refused (V15), no slot in them is refused as unfilled (V20), and the grader-side
    check (V17) sees the slot as consumed."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.validate import consumed, validate_state
    from refactory_packages.livemath.skill_graph import SkillGraph

    def issues(**knobs: str) -> list[tuple[str, str]]:
        state = _mode(**knobs).state_dict()
        return [(i.code, i.node) for i in validate_state(
            state, invariants=SkillGraph().invariants(), resolve_calls=True)]

    assert issues(decide_task="Reread {{user_prompt}} before you answer.") == []
    assert issues(decide_task="Use {{ghost}}.") == [("V15", "decide_prompt")]
    assert issues(compare_task="See {{decide_prompt}}.") == [("V15", "prepare")]
    state = _mode(decide_task="Use {{ghost}}.").state_dict()
    node = Graph.from_state(state["graph"]).nodes["decide_prompt"]
    values = {name: entry["value"] for name, entry in state["knobs"].items()}
    assert ("fact", "ghost") in consumed(node, values)
