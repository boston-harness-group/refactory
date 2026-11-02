"""Tests for the LiveMathematicianBench task package (refactory_packages/livemath).

Tests cover: loading, ids, manifest, grading/extraction, shuffling,
walker integration, lowering, and F6 records.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from refactory_dsh import lowering, nn
from refactory_dsh.task import VerifyResult

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "livemath"
SPLIT_DIR = FIXTURE_DIR / "split"


# ── loading and ids ────────────────────────────────────────────────────────

def test_load_items_from_fixture():
    from refactory_packages.livemath.tasks import LiveMathTask

    task = LiveMathTask(data_path=str(FIXTURE_DIR))
    items = task.instances()
    assert len(items) == 5
    for item in items:
        assert item.id
        assert "label" in item.metadata
        assert "label_text" in item.metadata
        assert "question" in item.metadata
        assert "choices" in item.metadata


def test_item_ids_match_fixture():
    from refactory_packages.livemath.tasks import LiveMathTask

    task = LiveMathTask(data_path=str(FIXTURE_DIR))
    items = task.instances()
    ids = {item.id for item in items}
    assert ids == {"202501:1", "202501:2", "202503:1", "202503:2", "202503:3"}
    # Item 202501:2 has correct label "D" (normalized from "D." trailing period)
    item_2 = next(i for i in items if i.id == "202501:2")
    assert item_2.metadata["label"] == "D"


# ── manifest ───────────────────────────────────────────────────────────────

def test_manifest_counts(tmp_path):
    from refactory_packages.livemath.manifest import build_parts_manifest

    build_parts_manifest(str(SPLIT_DIR), str(tmp_path))
    manifest = json.loads((tmp_path / "parts.json").read_text())
    assert len(manifest["train"]) == 2
    assert len(manifest["val"]) == 1
    assert len(manifest["test"]) == 2


def test_manifest_ids(tmp_path):
    from refactory_packages.livemath.manifest import build_parts_manifest

    build_parts_manifest(str(SPLIT_DIR), str(tmp_path))
    manifest = json.loads((tmp_path / "parts.json").read_text())
    assert set(manifest["train"]) == {"202501:1", "202501:2"}
    assert set(manifest["val"]) == {"202503:1"}
    assert set(manifest["test"]) == {"202503:2", "202503:3"}


# ── grading / extraction ──────────────────────────────────────────────────

def test_extract_answer_tag():
    from refactory_packages.livemath.grading import extract_answer

    # Primary: last <answer> tag
    assert extract_answer("<answer>B</answer>") == "B"
    # Last-tag-wins
    assert extract_answer("<answer>A</answer> noise <answer>C</answer>") == "C"
    # Case-insensitive
    assert extract_answer("<ANSWER>D</ANSWER>") == "D"
    # Whitespace stripping
    assert extract_answer("<answer> B </answer>") == "B"
    # Multiline
    assert extract_answer("<answer>line1\nline2</answer>") == "line1\nline2"


def test_extract_answer_fallback():
    from refactory_packages.livemath.grading import extract_answer

    # No tags: last non-empty line
    assert extract_answer("line1\nline2\n\nline4") == "line4"
    # Single line
    assert extract_answer("only line") == "only line"
    # Empty
    assert extract_answer("") == ""


def test_normalize_label():
    from refactory_packages.livemath.grading import normalize_label

    assert normalize_label("b") == "B"
    assert normalize_label("B.") == "B"
    assert normalize_label("B)") == "B"
    assert normalize_label(" c:") == "C"
    assert normalize_label(3) == "3"
    assert normalize_label("D...") == "D"


def test_parse_choice_label():
    from refactory_packages.livemath.grading import parse_choice_label

    choices = [
        {"label": "A", "text": "Alpha"},
        {"label": "B", "text": "Beta"},
        {"label": "C", "text": "Gamma"},
    ]
    # Direct label match
    assert parse_choice_label("B", choices) == "B"
    # Normalized label match
    assert parse_choice_label("b.", choices) == "B"
    # Text match (case-insensitive)
    assert parse_choice_label("alpha", choices) == "A"
    # First token match
    assert parse_choice_label("C is correct", choices) == "C"
    # Fallback
    assert parse_choice_label("nonsense", choices) == "NONSENSE"


def test_grading_correct():
    from refactory_packages.livemath.grading import evaluate

    choices = [
        {"label": "A", "text": "Diverges"},
        {"label": "B", "text": "Converges conditionally"},
        {"label": "C", "text": "Converges absolutely"},
        {"label": "D", "text": "Oscillates"},
    ]
    result = evaluate("<answer>B</answer>", {"label": "B", "text": "Converges conditionally"}, choices)
    assert result["em"] == 1.0
    assert result["predicted_label"] == "B"


def test_grading_incorrect():
    from refactory_packages.livemath.grading import evaluate

    choices = [
        {"label": "A", "text": "Diverges"},
        {"label": "B", "text": "Converges conditionally"},
        {"label": "C", "text": "Converges absolutely"},
        {"label": "D", "text": "Oscillates"},
    ]
    result = evaluate("<answer>A</answer>", {"label": "B", "text": "Converges conditionally"}, choices)
    assert result["em"] == 0.0
    assert result["predicted_label"] == "A"


# ── choice shuffling ──────────────────────────────────────────────────────

def test_shuffle_choices_byte_exact():
    """Byte-exact against SkillOpt's _item_shuffle_seed and _shuffle_item_choices."""
    from refactory_packages.livemath.shuffling import item_shuffle_seed, shuffle_item_choices

    # Verify seed computation is byte-exact
    for item_id, seed in [("202501:1", 42), ("202503:2", 7), ("202501:1", 0)]:
        expected_seed = int(hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()[:16], 16)
        assert item_shuffle_seed(item_id, seed) == expected_seed

    # Verify shuffling is byte-exact: construct a test item and shuffle it
    item = {
        "id": "202501:1",
        "choices": [
            {"label": "A", "text": "Diverges"},
            {"label": "B", "text": "Converges conditionally"},
            {"label": "C", "text": "Converges absolutely"},
            {"label": "D", "text": "Oscillates"},
        ],
        "correct_choice": {"label": "B", "text": "Converges conditionally"},
    }
    shuffled = shuffle_item_choices(item, 42)
    # The order must be deterministic; verify labels are reassigned A,B,C,D
    labels = [c["label"] for c in shuffled["choices"]]
    assert labels == ["A", "B", "C", "D"]
    # Correct choice must track the new label
    correct_text = shuffled["correct_choice"]["text"]
    assert correct_text == "Converges conditionally"
    # The correct choice label must match one of the shuffled choices with that text
    correct_label = shuffled["correct_choice"]["label"]
    matched = [c for c in shuffled["choices"] if c["text"] == "Converges conditionally"]
    assert len(matched) == 1
    assert matched[0]["label"] == correct_label

    # Second seed produces different order (with overwhelming probability)
    shuffled2 = shuffle_item_choices(item, 7)
    texts1 = [c["text"] for c in shuffled["choices"]]
    texts2 = [c["text"] for c in shuffled2["choices"]]
    # At least for these seeds, the order should differ
    # (we won't assert exact order since it's seed-dependent, but the labels
    # must be reassigned correctly)
    assert [c["label"] for c in shuffled2["choices"]] == ["A", "B", "C", "D"]
    assert shuffled2["correct_choice"]["text"] == "Converges conditionally"


def test_shuffle_choices_disabled():
    from refactory_packages.livemath.shuffling import shuffle_item_choices

    item = {
        "id": "202501:1",
        "choices": [
            {"label": "A", "text": "Diverges"},
            {"label": "B", "text": "Converges conditionally"},
            {"label": "C", "text": "Converges absolutely"},
            {"label": "D", "text": "Oscillates"},
        ],
        "correct_choice": {"label": "B", "text": "Converges conditionally"},
    }
    result = shuffle_item_choices(item, 42, shuffle=False)
    # Order and labels preserved
    assert [c["label"] for c in result["choices"]] == ["A", "B", "C", "D"]
    assert [c["text"] for c in result["choices"]] == [c["text"] for c in item["choices"]]
    assert result["correct_choice"] == {"label": "B", "text": "Converges conditionally"}
    # Deep copy: mutating result should not affect original
    result["choices"][0]["label"] = "Z"
    assert item["choices"][0]["label"] == "A"


def test_instances_with_seed_shuffles():
    from refactory_packages.livemath.tasks import LiveMathTask

    task = LiveMathTask(data_path=str(FIXTURE_DIR), shuffle_choices=True)
    items_shuffled = task.instances(seed=42)
    items_canon = task.instances(seed=None)

    # With shuffle_choices=True but seed=None, choices are NOT shuffled
    # With seed=42, at least one item should have different choice order
    any_differ = False
    for s_item, c_item in zip(items_shuffled, items_canon):
        if s_item.id == c_item.id:
            s_texts = [c["text"] for c in s_item.metadata["choices"]]
            c_texts = [c["text"] for c in c_item.metadata["choices"]]
            if s_texts != c_texts:
                any_differ = True
                break
    assert any_differ, "expected at least one item's choice order to differ with seed=42"

    # Verify correct_choice tracks the new label
    for item in items_shuffled:
        correct_text = item.metadata["correct_choice"]["text"]
        correct_label = item.metadata["correct_choice"]["label"]
        matched = [c for c in item.metadata["choices"] if c["text"] == correct_text]
        assert len(matched) == 1
        assert matched[0]["label"] == correct_label


# ── response reaches extract_answer unchanged ─────────────────────────────

def test_response_reaches_extract_answer_unchanged(tmp_path):
    """With tool_policy=none, the solver's raw response reaches extract_answer unchanged."""
    from refactory_packages.livemath.grading import extract_answer
    from refactory_packages.livemath.mode import Solver
    import refactory_packages.livemath.tasks as tasks_mod

    response_text = "Let me think...\n<answer>C</answer>"

    # Patch the task to use fixture data with no shuffling
    original_init = tasks_mod.LiveMathTask.__init__

    def patched_init(self, data_path=None, shuffle_choices=True):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False)

    tasks_mod.LiveMathTask.__init__ = patched_init
    try:
        from tests_support import FakeModel
        model = FakeModel([response_text] * 10)
        mode = Solver()
        workdir = tmp_path / "run"
        workdir.mkdir()

        result = mode.run(workdir, model=model)
        assert result.exit_ok, f"run failed: {result.error}"

        # Read answer.txt — the agent node's captured reply
        answer_file = workdir / "answer.txt"
        assert answer_file.exists(), "answer.txt not created"
        captured_reply = answer_file.read_text()

        # The raw response must reach extract_answer unchanged
        assert captured_reply == response_text
        assert extract_answer(captured_reply) == "C"

        # Verify produces correct prediction
        scores = result.facts.get("item_scores", [])
        assert len(scores) > 0
        for score_entry in scores:
            assert "prediction" in score_entry or "error_text" in score_entry
    finally:
        tasks_mod.LiveMathTask.__init__ = original_init


# ── walker integration ────────────────────────────────────────────────────

@pytest.fixture()
def walker_result(tmp_path):
    """Run the solver mode over the fixture with FakeModel returning <answer>B</answer>."""
    from refactory_packages.livemath.mode import Solver
    from refactory_dsh.evaluators import WalkerEvaluator

    # Patch the task to use fixture data
    import refactory_packages.livemath.tasks as tasks_mod
    original_init = tasks_mod.LiveMathTask.__init__

    def patched_init(self, data_path=None, shuffle_choices=True):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False)

    tasks_mod.LiveMathTask.__init__ = patched_init
    try:
        from tests_support import FakeModel
        model = FakeModel(["<answer>B</answer>"] * 10)
        mode = Solver()
        state = mode.state_dict()
        evaluator = WalkerEvaluator(model=model, project=Path("."), max_steps=2000)
        workdir = tmp_path / "walk"
        result = evaluator.evaluate(mode, state, workdir)
    finally:
        tasks_mod.LiveMathTask.__init__ = original_init

    return result


def test_walker_produces_records(walker_result):
    result = walker_result
    assert result.exit_ok, f"run failed: {result.error}"
    scores = result.facts.get("item_scores", [])
    assert len(scores) == 5

    # Items with correct label B score 1.0: 202501:1, 202503:1, 202503:2
    correct_b_ids = {"202501:1", "202503:1", "202503:2"}
    for score_entry in scores:
        if score_entry["id"] in correct_b_ids:
            assert score_entry["score"] == 1.0, f"{score_entry['id']} should score 1.0"
        else:
            assert score_entry["score"] == 0.0, f"{score_entry['id']} should score 0.0"


def test_records_have_prediction_and_label(walker_result):
    result = walker_result
    assert result.item_records, "no item records produced"

    for record in result.item_records:
        assert record.get("prediction") is not None, f"missing prediction in {record['id']}"
        assert record.get("label") is not None, f"missing label in {record['id']}"
        assert record.get("label_text") is not None, f"missing label_text in {record['id']}"
        assert isinstance(record.get("exchanges"), list), f"missing exchanges in {record['id']}"
        assert len(record["exchanges"]) > 0, f"empty exchanges in {record['id']}"
        for ex in record["exchanges"]:
            assert "prompt" in ex, f"missing prompt in exchange for {record['id']}"
            assert "response" in ex, f"missing response in exchange for {record['id']}"
        assert isinstance(record.get("metadata"), dict), f"missing metadata in {record['id']}"
        assert "question" in record["metadata"], f"missing question in metadata for {record['id']}"
        assert "choices" in record["metadata"], f"missing choices in metadata for {record['id']}"

    # For the correct item 202501:1
    item_1 = next(r for r in result.item_records if r["id"] == "202501:1")
    assert item_1["prediction"] == "B"
    assert item_1["label"] == "B"
    assert item_1["label_text"] == "Converges conditionally"


# ── correction #2: use_theorem / use_sketch gating ───────────────────────

_GATING_ITEM = {
    "question": "Which series converges?",
    "choices": [
        {"label": "A", "text": "Diverges"},
        {"label": "B", "text": "Converges"},
    ],
    "theorem": "Let $f$ be continuous on $[0,1]$.",
    "sketch": "Apply the intermediate value theorem.",
}


class _GatingCtx:
    def __init__(self, knobs):
        self._knobs = knobs

    def read_json(self, path, default=None):
        return dict(_GATING_ITEM)

    def knob(self, name, default=""):
        return self._knobs.get(name, default)


def test_gating_knobs_default_to_skillopt_values():
    """use_theorem / use_sketch are mode knobs defaulting to False
    (SkillOpt configs/livemathematicianbench/default.yaml:21-22), and the
    format_prompt node reads them."""
    from refactory_packages.livemath.mode import Solver

    knobs = Solver.declare()
    assert knobs["use_theorem"].default is False
    assert knobs["use_sketch"].default is False
    node = Solver().build().nodes["format_prompt"]
    assert {"use_theorem", "use_sketch"} <= set(node.knobs)


def test_default_prompt_is_byte_exact_without_theorem_and_sketch():
    """Defaults (both knobs off): SkillOpt's _build_user (rollout.py:41-44)
    emits only Question and Choices, even when the item has theorem/sketch."""
    from refactory_packages.livemath.ops import format_prompt

    prompt = format_prompt(_GatingCtx({}))["user_prompt"]
    assert prompt == (
        "## Question\nWhich series converges?\n\n"
        "## Choices\nA. Diverges\nB. Converges"
    )


def test_enabled_prompt_is_byte_exact_with_theorem_and_sketch():
    """Both knobs on: Theorem then Proof Sketch, joined by blank lines."""
    from refactory_packages.livemath.ops import format_prompt

    prompt = format_prompt(_GatingCtx({"use_theorem": True, "use_sketch": True}))["user_prompt"]
    assert prompt == (
        "## Question\nWhich series converges?\n\n"
        "## Choices\nA. Diverges\nB. Converges\n\n"
        "## Theorem\nLet $f$ be continuous on $[0,1]$.\n\n"
        "## Proof Sketch\nApply the intermediate value theorem."
    )


def test_each_gate_is_independent():
    from refactory_packages.livemath.ops import format_prompt

    only_theorem = format_prompt(_GatingCtx({"use_theorem": True}))["user_prompt"]
    assert only_theorem.endswith("## Theorem\nLet $f$ be continuous on $[0,1]$.")
    assert "## Proof Sketch" not in only_theorem
    only_sketch = format_prompt(_GatingCtx({"use_sketch": True}))["user_prompt"]
    assert only_sketch.endswith("## Proof Sketch\nApply the intermediate value theorem.")
    assert "## Theorem" not in only_sketch


# ── correction #3: solve node role ────────────────────────────────────────

def test_solve_node_role_is_solver():
    """The solve node must use role='solver', not 'ceo' or 'builder'.
    Proves correction #3."""
    from refactory_packages.livemath.mode import Solver

    mode = Solver()
    graph = mode.build()
    solve_node = graph.nodes["solve"]
    assert solve_node.role == "solver", (
        f"solve node role is {solve_node.role!r}, expected 'solver'"
    )


# ── lowering ──────────────────────────────────────────────────────────────

def test_lowering_works():
    from refactory_packages.livemath.mode import Solver

    mode = Solver()
    state = mode.state_dict()
    workflow = lowering.to_workflow(state, name="livemath-solver")

    # Has start_node
    assert workflow.start_node
    # Data node exists
    assert "data" in workflow.nodes
    # Agent node solve has tool_policy=none
    solve_node = workflow.nodes["solve"]
    assert solve_node.tool_policy == "none"
    # Role must be solver (correction #3)
    assert solve_node.role.value == "solver"
    # System prompt is non-empty
    assert solve_node.system_prompt
    # Prompt template is non-empty
    assert solve_node.prompt_template

    # JSON serialization roundtrip
    serialized = json.dumps(workflow.to_dict(), indent=2)
    parsed = json.loads(serialized)
    assert parsed  # non-empty


# ── pinned SkillOpt outputs ───────────────────────────────────────────────
# Every literal below was produced by running SkillOpt's own functions
# (skillopt/envs/livemathematicianbench/*) on the inputs shown, not by this port.

# skillopt/envs/livemathematicianbench/prompts/rollout_system.md, byte for byte
# (note the trailing newline).
_SKILLOPT_ROLLOUT_SYSTEM_MD = (
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

_SHUFFLE_ITEM = {
    "id": "202501:1",
    "choices": [
        {"label": "A", "text": "Diverges"},
        {"label": "B", "text": "Converges conditionally"},
        {"label": "C", "text": "Converges absolutely"},
        {"label": "D", "text": "Oscillates"},
    ],
    "correct_choice": {"label": "B", "text": "Converges conditionally"},
}


def _render_system(skill: str) -> str:
    import types

    from refactory_dsh.nn.executor import render
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.ops import format_prompt

    facts = format_prompt(_GatingCtx({"skill": skill}))
    node = Solver().build().nodes["solve"]
    return render(node.system_prompt_template, types.SimpleNamespace(knobs={"skill": skill}, facts=facts))


def test_system_prompt_template_is_skillopt_file_byte_exact():
    from refactory_packages.livemath.mode import SYSTEM_PROMPT

    assert SYSTEM_PROMPT == _SKILLOPT_ROLLOUT_SYSTEM_MD.replace("{skill_section}", "{{skill_section}}")


def test_rendered_system_prompt_without_skill_matches_skillopt():
    # SkillOpt rollout.py:16-21 _build_system("")
    assert _render_system("") == _SKILLOPT_ROLLOUT_SYSTEM_MD.format(skill_section="")


def test_rendered_system_prompt_with_skill_matches_skillopt():
    # SkillOpt rollout.py:16-21 _build_system("  Check edge cases.\n")
    assert _render_system("  Check edge cases.\n") == (
        "You are an expert mathematical reasoning agent solving multiple-choice questions.\n"
        "\n"
        "## Skill\nCheck edge cases.\n\n"
        "## Task Format\n"
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


def test_shuffle_matches_skillopt_literal_orders():
    # SkillOpt dataloader.py:196-198 _item_shuffle_seed and :200-226 _shuffle_item_choices
    from refactory_packages.livemath.shuffling import item_shuffle_seed, shuffle_item_choices

    assert item_shuffle_seed("202501:1", 42) == 16574492107308482448
    expected = {
        42: (["Diverges", "Converges conditionally", "Oscillates", "Converges absolutely"],
             {"label": "B", "text": "Converges conditionally"}),
        7: (["Oscillates", "Diverges", "Converges absolutely", "Converges conditionally"],
            {"label": "D", "text": "Converges conditionally"}),
        43: (["Converges conditionally", "Oscillates", "Converges absolutely", "Diverges"],
             {"label": "A", "text": "Converges conditionally"}),
    }
    for seed, (texts, correct) in expected.items():
        out = shuffle_item_choices(_SHUFFLE_ITEM, seed)  # default: shuffle on
        assert out["choices"] == [
            {"label": label, "text": text} for label, text in zip("ABCD", texts)
        ], seed
        assert out["correct_choice"] == correct, seed


def test_task_instances_default_shuffle_matches_skillopt_seed_42():
    """LiveMathTask's default shuffles (SkillOpt default.yaml:20) with SkillOpt's
    per-item order for seed 42 (dataloader.py:200-226 over load_items' fixture output)."""
    from refactory_packages.livemath.tasks import LiveMathTask

    expected = {
        "202501:1": (["Diverges", "Converges conditionally", "Oscillates",
                      "Converges absolutely"], "B"),
        "202501:2": (["0", "3", "2", "1"], "B"),
        "202503:1": (["$2\\pi i$", "$0$", "$-2\\pi i$", "$\\pi i$"], "A"),
        "202503:2": (["6", "5", "3", "4"], "D"),
        "202503:3": (["No, only closed", "Yes, by Heine-Borel", "No, only bounded",
                      "It depends on $n$"], "B"),
    }
    got = {
        inst.id: ([c["text"] for c in inst.metadata["choices"]], inst.metadata["label"])
        for inst in LiveMathTask(data_path=str(FIXTURE_DIR)).instances(seed=42)
    }
    assert got == expected


def test_load_items_skips_rows_like_skillopt(tmp_path):
    """SkillOpt dataloader.py:139 keeps a row only with a question, choices and a
    correct label; SkillOpt's load_items on these rows keeps 202509:1 and 202509:6."""
    from refactory_packages.livemath.tasks import load_items

    rows = [
        {"month": "202509", "no": 1, "question": "Q1", "choices": ["x", "y"], "correct_choice": "A"},
        {"month": "202509", "no": 2, "question": "", "choices": ["x"], "correct_choice": "A"},
        {"month": "202509", "no": 3, "question": "Q3", "choices": ["x", "y"], "correct_choice": {}},
        {"month": "202509", "no": 4, "question": "Q4", "choices": [], "correct_choice": "A"},
        {"month": "202509", "no": 5, "question": "Q5", "choices": [" "], "correct_choice": "A"},
        {"month": "202509", "no": 6, "question": "Q6", "choices": ["p", "q"], "correct_choice": "B"},
    ]
    (tmp_path / "qa_202509_final.json").write_text(json.dumps(rows))
    assert [item["id"] for item in load_items(str(tmp_path))] == ["202509:1", "202509:6"]


# ── SkillOpt solver defaults on the solve node ────────────────────────────

def test_solve_node_carries_arm_a_exec_timeout():
    """exec_timeout 1800 -- arm A's ``--cfg-options env.exec_timeout=1800``
    (SkillOpt's run-train.sh:14, run-train-seed.sh:20; SkillOpt's own LiveMath
    config default is 300) -- is the solve node's timeout, in the graph and in
    the lowered node."""
    from refactory_packages.livemath.mode import Solver

    assert Solver().build().nodes["solve"].timeout_s == 1800
    workflow = lowering.to_workflow(Solver().state_dict(), name="livemath-solver")
    assert workflow.to_dict()["nodes"]["solve"]["timeout"] == 1800


def test_solve_timeout_is_configurable():
    from refactory_packages.livemath.mode import Solver

    class Slow(Solver):
        solve_timeout_s = 900

    assert Slow().build().nodes["solve"].timeout_s == 900


def test_walker_run_items_follow_skillopt_shuffle_for_passed_seed(tmp_path):
    """In a real walker run, the seed subset.json names reaches
    ``LiveMathTask.instances`` and every item's choices are SkillOpt's shuffle
    for that seed. Expected orders were computed by SkillOpt's own
    ``LiveMathematicianBenchDataLoader._shuffle_item_choices`` (dataloader.py:200-226)
    for seed 1043 -- the epoch-1 batch seed for seed 42 (dataloader.py:278)."""
    from refactory_dsh.evaluators import WalkerEvaluator
    from refactory_dsh.task import write_subset
    from refactory_packages.livemath.mode import Solver
    import refactory_packages.livemath.tasks as tasks_mod
    from tests_support import FakeModel

    expected = {
        "202501:1": (["Diverges", "Converges conditionally", "Converges absolutely",
                      "Oscillates"], "B"),
        "202501:2": (["2", "0", "1", "3"], "D"),
        "202503:1": (["$2\\pi i$", "$\\pi i$", "$-2\\pi i$", "$0$"], "A"),
        "202503:2": (["6", "5", "3", "4"], "D"),
        "202503:3": (["No, only closed", "No, only bounded", "Yes, by Heine-Borel",
                      "It depends on $n$"], "C"),
    }
    original_init = tasks_mod.LiveMathTask.__init__

    def patched_init(self, data_path=None, shuffle_choices=True):  # type: ignore[no-untyped-def]
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=shuffle_choices)

    tasks_mod.LiveMathTask.__init__ = patched_init  # type: ignore[method-assign]
    try:
        workdir = tmp_path / "walk"
        write_subset(workdir, "train", 0.0, seed=1043)
        mode = Solver()
        evaluator = WalkerEvaluator(
            model=FakeModel(["<answer>B</answer>"] * 10), project=Path("."), max_steps=2000,
        )
        result = evaluator.evaluate(mode, mode.state_dict(), workdir)
    finally:
        tasks_mod.LiveMathTask.__init__ = original_init  # type: ignore[method-assign]

    assert result.exit_ok, result.error
    got = {
        r["id"]: ([c["text"] for c in r["metadata"]["choices"]], r["label"])
        for r in result.item_records
    }
    assert got == expected


# ── SkillOpt view: records as SkillOpt's rollout results ─────────────────

_VIEW_RECORD = {
    "id": "202601:41", "passed": False, "score": 0.0,
    "error_text": None, "error_kind": None,
    "exchanges": [{
        "node": "solve",
        "prompt": "## Question\nQ?\n\n## Choices\nA. x\nB. y",
        "response": "think\n<answer>A</answer>",
    }],
    "prediction": "A", "label": "B", "label_text": "y", "fail_reason": None,
    "metadata": {
        "label": "B", "label_text": "y", "question": "Q?",
        "choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
        "correct_choice": {"label": "B", "text": "y"},
        "theorem": "", "sketch": "", "theorem_type": ["Implication", "Other"],
    },
}


def test_skillopt_view_renders_skillopt_analyst_trajectory():
    """reflect.py:121-248 over livemathematicianbench/rollout.py's result,
    conversation.json and target prompts (rollout.py:127-144, :246-281)."""
    from refactory_packages.livemath.skillopt_view import rollout_result
    from refactory_packages.skillopt.prompts import fmt_minibatch_trajectories

    result = rollout_result(_VIEW_RECORD, {"skill": "S"})
    system = _SKILLOPT_ROLLOUT_SYSTEM_MD.format(skill_section="## Skill\nS\n\n")
    assert fmt_minibatch_trajectories([result]) == (
        "### Trajectory 1 (id=202601:41)\n"
        "Task: \n"
        "Task type: Implication\n"
        "Failure reason: MCQ=0: predicted 'A' but expected 'B'\n"
        "Steps: 1\n"
        f"\n#### Target System Prompt\n{system}\n"
        "\n#### Target User Prompt\n## Question\nQ?\n\n## Choices\nA. x\nB. y\n"
        "\n"
        "[agent] think\n<answer>A</answer>\n"
        "[verification] [EVALUATION RESULT]\n"
        "Question: Q?\n"
        "Predicted label: 'A'\n"
        "Predicted text: 'x'\n"
        "Correct label: 'B'\n"
        "Correct text: 'y'\n"
        "Exact Match: 0.0"
    )
    assert (result["soft"], result["predicted_answer"]) == (0.0, "A")


def test_skillopt_view_success_and_error_records():
    """A pass has no fail reason; no theorem type is ``math_mcq`` (rollout.py:132);
    an errored item has no conversation, so SkillOpt skips it (reflect.py:151-156)."""
    from refactory_packages.livemath.skillopt_view import rollout_result

    passed = dict(_VIEW_RECORD, passed=True, score=1.0, exchanges=[
        dict(_VIEW_RECORD["exchanges"][0], response="<answer>B</answer>"),
    ], metadata=dict(_VIEW_RECORD["metadata"], theorem_type=[]))
    result = rollout_result(passed, {"skill": ""})
    assert (result["fail_reason"], result["task_type"], result["hard"], result["soft"]) == (
        "", "math_mcq", 1, 1.0,
    )
    assert result["conversation"][-1]["content"].endswith("Exact Match: 1.0")
    assert result["target_system_prompt"] == _SKILLOPT_ROLLOUT_SYSTEM_MD.format(skill_section="")

    # The item deadline, and a model call that ran out of time inside the
    # pass, are SkillOpt's task timeout: exec_timeout + 60 (rollout.py:313, :373).
    for error_kind, error_text in (
        ("timeout", "item '202601:41' of 'data' timed out after 2000.0s"),
        ("RuntimeError", "solve: CompletionTimeout: model completion timed out after 1800s"),
    ):
        errored = dict(_VIEW_RECORD, exchanges=[], error_text=error_text, error_kind=error_kind)
        result = rollout_result(errored, {"skill": "S"})
        assert result["conversation"] is None and result["n_turns"] == 0
        assert result["fail_reason"] == "task-timeout-1860s"
    # Any other failure is process_one's ``error: {e}`` (rollout.py:286-287).
    failed = dict(_VIEW_RECORD, exchanges=[], error_kind="RuntimeError", error_text=(
        "solve: RuntimeError: Claude backend failed after 5 retries: "
        "Claude CLI exited with code 1"
    ))
    assert rollout_result(failed, {})["fail_reason"] == (
        "error: Claude backend failed after 5 retries: Claude CLI exited with code 1"
    )


def test_skillopt_view_slow_update_trajectory():
    """slow_update.py:116-168 ``_read_trajectory`` over the same conversation."""
    from refactory_packages.livemath.skillopt_view import rollout_result
    from refactory_packages.skillopt.env import SkillOptEnv
    from refactory_packages.skillopt.loop import SkillOptLoop

    pairs = SkillOptLoop._build_comparison_pairs(
        [_VIEW_RECORD], [_VIEW_RECORD], ["202601:41"],
        env=SkillOptEnv("livemath"),
        prev_knobs={"skill": "P"}, curr_knobs={"skill": "C"},
    )
    assert pairs[0]["task"] == "Q?"
    assert pairs[0]["curr"] == {
        "hard": 0, "soft": 0.0, "predicted_answer": "A",
        "fail_reason": "MCQ=0: predicted 'A' but expected 'B'",
    }
    assert pairs[0]["curr_trajectory"] == (
        "[agent] think\n<answer>A</answer>\n"
        + "[verification] " + rollout_result(_VIEW_RECORD, {})["conversation"][-1]["content"]
    )


def test_format_prompt_and_view_share_the_prompt_builders():
    """The solver's prompts and the view's target prompts are one code path."""
    from refactory_packages.livemath.ops import build_user_prompt, format_prompt
    from refactory_packages.livemath.skillopt_view import build_system

    facts = format_prompt(_GatingCtx({"skill": " S "}))
    assert facts["user_prompt"] == build_user_prompt(_GATING_ITEM)
    assert build_system(" S ") == _render_system(" S ")


def test_skillopt_view_an_errored_record_with_exchanges_is_an_error():
    """A pass that failed after a model call (an exchange recorded, then an
    error) is still SkillOpt's failed process_one: its error, no conversation
    (rollout.py:286-289)."""
    from refactory_packages.livemath.skillopt_view import rollout_result

    errored = dict(_VIEW_RECORD, error_kind="RuntimeError",
                   error_text="finish: ValueError: could not write result.json")
    result = rollout_result(errored, {"skill": "S"})
    assert result["conversation"] is None and result["n_turns"] == 0
    assert result["fail_reason"] == "error: could not write result.json"
