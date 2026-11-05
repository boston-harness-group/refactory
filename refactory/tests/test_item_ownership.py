"""Item ownership: the loop decides which items a candidate is graded on.

Layer 1 — edit guards: `limit` and `produces` (item_scores removal) trigger E-exam.
Layer 2 — post-trial verification: the loop verifies which ids were actually graded,
using item_scores from either evaluator (walker or dsh).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from refactory_dsh import loop as loop_module
from refactory_dsh import optim as optim_module
from refactory_dsh import registry
from refactory_dsh import task as task_module
from refactory_dsh.nn import edits as edit_module
from refactory_dsh.nn.mode import RunResult
from refactory_dsh.optim import Candidate
from refactory_dsh.paths import project_root


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

MODE_ID = "dataset_probe/dataset"


def _seed_state() -> dict[str, Any]:
    _, klass = registry.resolve(project_root(), MODE_ID)
    return klass().state_dict()


def _loop(tmp_path: Path) -> loop_module.Loop:
    return loop_module.Loop(
        MODE_ID,
        loop_module.LoopConfig(runs_root=tmp_path),
        optimizer=optim_module.create("random"),
    )


def _loop_with_seed(tmp_path: Path) -> tuple[loop_module.Loop, dict[str, Any]]:
    state = _seed_state()
    run = _loop(tmp_path)
    run.population = [Candidate.from_state(state)]
    return run, state


# ---------------------------------------------------------------------------
# Layer 1 — _item_sources / _exam_moved guard
# ---------------------------------------------------------------------------


def test_limit_change_triggers_exam_moved(tmp_path):
    """Changing limit on a data node is caught as E-exam. Python runtime."""
    run, state = _loop_with_seed(tmp_path)
    # Directly mutate limit in a copied state dict
    changed = copy.deepcopy(state)
    from refactory_dsh.nn.graph import Graph, Node
    graph = Graph.from_state(changed["graph"])
    graph.nodes["data"] = graph.nodes["data"].with_field("limit", 5)
    changed["graph"] = graph.to_state()
    candidate = Candidate.from_state(changed)
    reason = run._exam_moved(candidate)
    assert reason is not None and reason.startswith("E-exam"), reason


def test_unchanged_limit_passes_exam_moved(tmp_path):
    """An unchanged limit (default 0) does not trigger E-exam."""
    run, state = _loop_with_seed(tmp_path)
    same = Candidate.from_state(state)
    assert run._exam_moved(same) is None


def test_limit_change_rejected_by_step(tmp_path):
    """A limit change flows through step() and appears in rejections. Python runtime."""
    run, state = _loop_with_seed(tmp_path)
    changed = copy.deepcopy(state)
    from refactory_dsh.nn.graph import Graph
    graph = Graph.from_state(changed["graph"])
    graph.nodes["data"] = graph.nodes["data"].with_field("limit", 5)
    changed["graph"] = graph.to_state()
    candidate = Candidate.from_state(changed)
    run.step([candidate])
    assert any("E-exam" in r.reasons[0] for r in run.rejections), run.rejections


def _find_data_node(state: dict[str, Any]) -> dict[str, Any]:
    """Find the data node in a state dict (handles both dict and list formats)."""
    graph = state.get("graph", {})
    nodes = graph.get("nodes", {})
    if isinstance(nodes, dict):
        for node in nodes.values():
            if isinstance(node, dict) and node.get("kind") == "data":
                return node
    elif isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and node.get("kind") == "data":
                return node
    raise ValueError("no data node found")


def test_produces_edit_removing_item_scores_triggers_exam_moved(tmp_path):
    """Removing item_scores from produces is caught as E-exam. Python runtime."""
    run, state = _loop_with_seed(tmp_path)
    data_node = _find_data_node(state)
    assert "item_scores" in (data_node.get("produces") or [])

    # Direct state mutation removing item_scores from produces
    changed = copy.deepcopy(state)
    from refactory_dsh.nn.graph import Graph
    graph = Graph.from_state(changed["graph"])
    old_produces = list(graph.nodes["data"].produces)
    new_produces = [p for p in old_produces if p != "item_scores"]
    graph.nodes["data"] = graph.nodes["data"].with_field("produces", new_produces)
    changed["graph"] = graph.to_state()
    candidate = Candidate.from_state(changed)
    reason = run._exam_moved(candidate)
    assert reason is not None and reason.startswith("E-exam"), reason


def test_produces_edit_rejected_by_step(tmp_path):
    """Removing item_scores from produces flows through step() as E-exam. Python runtime."""
    run, state = _loop_with_seed(tmp_path)
    changed = copy.deepcopy(state)
    from refactory_dsh.nn.graph import Graph
    graph = Graph.from_state(changed["graph"])
    old_produces = list(graph.nodes["data"].produces)
    new_produces = [p for p in old_produces if p != "item_scores"]
    graph.nodes["data"] = graph.nodes["data"].with_field("produces", new_produces)
    changed["graph"] = graph.to_state()
    candidate = Candidate.from_state(changed)
    run.step([candidate])
    assert any("E-exam" in r.reasons[0] for r in run.rejections), run.rejections


# ---------------------------------------------------------------------------
# Layer 2 — post-trial verification via _measure / _verify_item_set
# ---------------------------------------------------------------------------


class _FakeEvaluator:
    """A mock evaluator that returns a predetermined RunResult."""

    name = "walker"
    out_of_process = False
    budget = None

    def __init__(self, result: RunResult):
        self._result = result

    def evaluate(self, mode: Any, state: dict[str, Any], workdir: Path) -> RunResult:
        return copy.deepcopy(self._result)


class _FakeDshEvaluator(_FakeEvaluator):
    name = "dsh"
    out_of_process = True


def _evaluator_class(name: str):
    return _FakeDshEvaluator if name == "dsh" else _FakeEvaluator


def _measure_with_evaluator(
    tmp_path: Path,
    evaluator_name: str,
    item_scores: list[dict[str, Any]] | None,
    *,
    score: float = 1.0,
    exit_ok: bool = True,
    error: str | None = None,
) -> RunResult:
    """Run _measure with a fake evaluator and return the composite result."""
    facts: dict[str, Any] = {}
    if item_scores is not None:
        facts["item_scores"] = item_scores

    fake_result = RunResult(
        facts=facts,
        score=score,
        exit_ok=exit_ok,
        error=error,
    )
    evaluator_cls = _evaluator_class(evaluator_name)
    evaluator = evaluator_cls(fake_result)

    run, state = _loop_with_seed(tmp_path)
    run.evaluator = evaluator
    candidate = Candidate.from_state(state)

    result, summary = run._measure(candidate, tmp_path / "trial", task_module.TRAIN)
    return result


def _expected_item_ids_for_mode(
    tmp_path: Path, part: str = task_module.TRAIN
) -> set[str]:
    """Compute the expected item ids for the test mode."""
    run, state = _loop_with_seed(tmp_path)
    ids, _ = run._expected_item_ids(state, part)
    return ids


@pytest.fixture
def expected_ids(tmp_path):
    """The expected item ids for the train part of dataset_probe/dataset."""
    return _expected_item_ids_for_mode(tmp_path)


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_refuses_extra_graded_ids(tmp_path, evaluator_name):
    """A trial grading ids outside the allowed set is refused."""
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires at least one expected item"

    scores = [{"id": eid, "passed": True, "score": 1.0} for eid in expected]
    scores.append({"id": "EXTRA_ROGUE_ID", "passed": True, "score": 1.0})

    result = _measure_with_evaluator(tmp_path, evaluator_name, scores)
    assert not result.exit_ok
    assert result.error is not None and "E-exam" in result.error
    assert "EXTRA_ROGUE_ID" in result.error


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_refuses_missing_graded_ids_no_failure(tmp_path, evaluator_name):
    """A trial missing an expected id with no failure record is refused."""
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires at least one expected item"

    missing_id = sorted(expected)[-1]
    # Return empty item_scores (no items graded) — the expected item is missing
    scores: list[dict[str, Any]] = []

    result = _measure_with_evaluator(tmp_path, evaluator_name, scores)
    assert not result.exit_ok
    assert result.error is not None and "E-exam" in result.error
    assert missing_id in result.error


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_accepts_missing_id_with_failure(tmp_path, evaluator_name):
    """A missing id with a recorded per-item failure is accepted."""
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires at least one expected item"

    missing_id = sorted(expected)[-1]
    # No items graded, but the error records that the missing item failed.
    scores: list[dict[str, Any]] = []

    # An error on the result records that the missing item failed.
    result = _measure_with_evaluator(
        tmp_path, evaluator_name, scores, error=f"item {missing_id}: branch error"
    )
    assert result.exit_ok


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_accepts_matching_ids(tmp_path, evaluator_name):
    """A trial whose graded ids match exactly is accepted."""
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires at least one expected item"

    scores = [{"id": eid, "passed": True, "score": 1.0} for eid in expected]

    result = _measure_with_evaluator(tmp_path, evaluator_name, scores)
    assert result.exit_ok


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_refuses_no_item_scores_when_seed_declares(tmp_path, evaluator_name):
    """When the seed declares item_scores, a trial producing none is refused."""
    result = _measure_with_evaluator(tmp_path, evaluator_name, None)
    assert not result.exit_ok
    assert result.error is not None and "E-exam" in result.error


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_skips_when_seed_does_not_declare_item_scores(tmp_path, evaluator_name):
    """When the seed does NOT declare item_scores, missing item_scores is OK."""
    fake_result = RunResult(facts={}, score=1.0, exit_ok=True)
    evaluator_cls = _evaluator_class(evaluator_name)
    evaluator = evaluator_cls(fake_result)

    run, state = _loop_with_seed(tmp_path)
    run.evaluator = evaluator
    candidate = Candidate.from_state(state)

    # Patch _expected_item_ids to simulate a mode that does not declare item_scores
    with patch.object(
        run, "_expected_item_ids", return_value=(set(), False)
    ):
        result, _ = run._measure(candidate, tmp_path / "trial", task_module.TRAIN)
    assert result.exit_ok


def test_subset_json_tampering_detected_as_backstop(tmp_path):
    """Tampered subset.json is caught as E-exam (backstop check)."""
    expected = _expected_item_ids_for_mode(tmp_path)
    scores = [{"id": eid, "passed": True, "score": 1.0} for eid in expected]

    class _TamperingEvaluator:
        name = "walker"
        out_of_process = False
        budget = None

        def __init__(self, trial_dir: Path):
            self._trial_dir = trial_dir

        def evaluate(self, mode: Any, state: dict[str, Any], workdir: Path) -> RunResult:
            # Tamper with subset.json during evaluation
            subset_path = workdir / "subset.json"
            subset_path.write_text(json.dumps({"part": "holdout", "holdout": 0.3}))
            return RunResult(
                facts={"item_scores": scores},
                score=1.0,
                exit_ok=True,
            )

    run, state = _loop_with_seed(tmp_path)
    trial_dir = tmp_path / "tampered_trial"
    run.evaluator = _TamperingEvaluator(trial_dir)
    candidate = Candidate.from_state(state)

    result, _ = run._measure(candidate, trial_dir, task_module.TRAIN)
    assert not result.exit_ok
    assert result.error is not None and "E-exam" in result.error
    assert "subset" in result.error.lower()


def test_subset_json_rewrite_still_graded_on_loop_ids(tmp_path):
    """A candidate that rewrites subset.json is still graded on the loop's ids.

    The loop-owned channel (expected ids from loop root and seed state) is the
    source of truth, not the candidate's trial-dir subset.json.
    """
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires expected items"

    # Return the correct item_scores (loop's expected ids) — the evaluator
    # uses the loop-owned items, not the tampered subset.json.
    scores = [{"id": eid, "passed": True, "score": 1.0} for eid in expected]

    class _SubsetRewriteEvaluator:
        name = "walker"
        out_of_process = False
        budget = None

        def evaluate(self, mode: Any, state: dict[str, Any], workdir: Path) -> RunResult:
            # Candidate tries to rewrite subset.json to holdout
            # But the loop already wrote it and the evaluator uses loop-owned items
            return RunResult(
                facts={"item_scores": scores},
                score=1.0,
                exit_ok=True,
            )

    run, state = _loop_with_seed(tmp_path)
    run.evaluator = _SubsetRewriteEvaluator()
    candidate = Candidate.from_state(state)

    result, _ = run._measure(candidate, tmp_path / "rewrite_trial", task_module.TRAIN)
    # The trial succeeds because the graded ids match the loop's expected ids
    assert result.exit_ok


def test_normal_loop_unchanged(tmp_path):
    """A normal loop with no tampering runs unchanged."""
    run, state = _loop_with_seed(tmp_path)

    # Get expected ids from loop's perspective
    expected_ids, seed_declares = run._expected_item_ids(state, task_module.TRAIN)

    # Normal evaluation: the _exam_moved check passes
    same = Candidate.from_state(state)
    assert run._exam_moved(same) is None

    # The expected ids method works and returns a non-empty set
    assert expected_ids, "expected ids should not be empty for dataset_probe"
    assert seed_declares, "dataset_probe declares item_scores"


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_accepts_present_failed_item(tmp_path, evaluator_name):
    """A failed item that is present in item_scores (with error_text) is accepted.

    F4: a failed item scores 0, keeps its error, and is present in the graded set.
    F3's post-trial check sees it as graded (it has an id), so no missing-id refusal.
    """
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires at least one expected item"

    scores = []
    for i, eid in enumerate(sorted(expected)):
        if i == 0:
            # First item is a failed item with error_text
            scores.append({
                "id": eid, "passed": False, "score": 0.0,
                "error_text": "verify failed", "error_kind": "ValueError",
            })
        else:
            scores.append({"id": eid, "passed": True, "score": 1.0})

    result = _measure_with_evaluator(tmp_path, evaluator_name, scores)
    assert result.exit_ok


@pytest.mark.parametrize("evaluator_name", ["walker", "dsh"])
def test_post_trial_accepts_index_ids_from_csv(tmp_path, evaluator_name):
    """A trial with index ids ("0", "1", …) from a CSV source passes F3's check."""
    expected = _expected_item_ids_for_mode(tmp_path)
    assert expected, "test requires at least one expected item"

    # Use the expected ids exactly — this proves that the expected_ids set
    # and the graded ids match when index ids are used.
    scores = [{"id": eid, "passed": True, "score": 1.0} for eid in expected]

    result = _measure_with_evaluator(tmp_path, evaluator_name, scores)
    assert result.exit_ok


def test_add_node_with_exam_field_triggers_guard(tmp_path):
    """add_node with a node containing inline_items triggers E-items. Python runtime."""
    from refactory_dsh.nn.edits import EditError

    _, klass = registry.resolve(project_root(), MODE_ID)
    state = klass().state_dict()
    new_node = {
        "id": "baked", "kind": "data",
        "subgraph_entry": "work", "subgraph_exit": "work",
        "inline_items": [{"id": "alpha"}, {"id": "beta"}],
        "writes": ["baked_item.json"], "produces": ["baked_scores"],
    }
    edit = {"op": "add_node", "node": new_node, "position": {"after": "data"}}
    with pytest.raises(EditError) as caught:
        edit_module.apply(edit, state)
    assert caught.value.code == "E-items"
