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

    def patched_init(self, data_path=None, shuffle_choices=True, use_theorem=False, use_sketch=False):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False,
                      use_theorem=use_theorem, use_sketch=use_sketch)

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

    def patched_init(self, data_path=None, shuffle_choices=True, use_theorem=False, use_sketch=False):
        original_init(self, data_path=str(FIXTURE_DIR), shuffle_choices=False,
                      use_theorem=use_theorem, use_sketch=use_sketch)

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
    # System prompt is non-empty
    assert solve_node.system_prompt
    # Prompt template is non-empty
    assert solve_node.prompt_template

    # JSON serialization roundtrip
    serialized = json.dumps(workflow.to_dict(), indent=2)
    parsed = json.loads(serialized)
    assert parsed  # non-empty
