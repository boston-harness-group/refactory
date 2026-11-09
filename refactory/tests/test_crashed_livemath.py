"""A crashed step in the real LiveMath modes: the item is scored or has no answer, never
an error naming the step after it.

A step whose retries are spent is missing (`test_crashed_step_retry`). A later op that
needs its output -- a read, a required input -- used to fail the item with an error that
named that op. Now the item has no answer (score 0, no error, ``missing_steps`` naming the
crashed step and why), and an op that combines a fork's branches (the judge's
``merge_attempts``, skill_graph's ``decide_prompt``) goes on with the branch that answered.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, workers
from refactory_dsh.model import CompletionFailed

from test_crashed_step_retry import OUTPUT_LIMIT, PASS_RETRY_TIMES, WORKER_CRASH, Scripted, needs_spine
from test_livemath_blocks import DRAIN_AGENT, FIXTURE_DIR, NODE, PYTHON_BIN, REPO_ROOT, _patch_task

SESSIONS = 1 + PASS_RETRY_TIMES
ITEMS = 5  # the LiveMath fixture


class _Crashing:
    """A walker model keyed by node: the nodes in ``crash`` hit the output limit every
    session; the others answer B. Calls and prompts are shared by every per-node copy."""

    def __init__(self, crash: set[str], log: list[tuple[str, str]] | None = None, node: str = ""):
        self.crash, self.log, self.node = crash, ([] if log is None else log), node

    def for_node(self, node_id: str, role: str = "") -> "_Crashing":
        return _Crashing(self.crash, self.log, node_id)

    def with_tool_policy(self, policy: str) -> "_Crashing":
        return self

    def with_wall(self, wall: Any) -> "_Crashing":
        return self

    def with_budget(self, budget: Any) -> "_Crashing":
        return self

    def context(self) -> dict:
        return {}

    def route(self) -> dict:
        return {"worker_backend": "dsh"}

    def complete(self, prompt: str, **_: Any) -> str:
        self.log.append((self.node, prompt))
        if self.node in self.crash:
            raise CompletionFailed(f"model completion failed (exit 1): {OUTPUT_LIMIT}",
                                   kind="output_limit")
        return "<answer>B</answer>"

    def calls(self, node: str) -> int:
        return sum(1 for name, _ in self.log if name == node)

    def prompts(self, node: str) -> list[str]:
        return [prompt for name, prompt in self.log if name == node]


def _mode(name: str) -> Any:
    from refactory_packages import livemath

    return next(cls for cls in livemath.PACKAGE.modes if cls.name == name)()


def _walk(name: str, crash: set[str], tmp_path: Path, monkeypatch) -> tuple[list[dict], _Crashing]:
    _patch_task(monkeypatch)
    model = _Crashing(crash)
    run = tmp_path / "walker"
    run.mkdir()
    result = _mode(name).run(run, model=model)
    assert result.exit_ok, result.error
    return result.facts["item_scores"], model


def _no_answer(records: list[dict], steps: set[str]) -> None:
    assert len(records) == ITEMS
    for record in records:
        assert (record["passed"], record["score"], record["error_kind"], record["error_text"]) \
            == (False, 0.0, None, None), record
        assert set(record["missing_steps"]) == steps
        assert record["fail_reason"].startswith("no answer: ")
        for info in record["missing_steps"].values():
            assert info["sessions"] == SESSIONS


def _answered(records: list[dict], missing: set[str]) -> None:
    assert len(records) == ITEMS
    for record in records:
        assert record["error_kind"] is None and record["error_text"] is None, record
        assert not str(record.get("fail_reason") or "").startswith("no answer"), record
        assert set(record.get("missing_steps") or {}) == missing


# ── the walker ─────────────────────────────────────────────────────────────


def test_judge_goes_on_with_the_attempt_that_answered(tmp_path, monkeypatch):
    """[walker] solver_judge: solve_a hits the output limit through its retries; the judge
    sees solve_b's attempt and a no-answer note for A, and every item is graded."""
    from refactory_packages.livemath.block_ops import NO_ATTEMPT

    records, model = _walk("solver_judge", {"solve_a"}, tmp_path, monkeypatch)
    _answered(records, {"solve_a"})
    assert model.calls("solve_a") == ITEMS * SESSIONS and model.calls("judge") == ITEMS
    assert all(NO_ATTEMPT in prompt for prompt in model.prompts("judge"))
    assert records[0]["missing_steps"]["solve_a"]["cause"] == "output_limit"


def test_judge_with_no_attempt_has_no_answer(tmp_path, monkeypatch):
    """[walker] solver_judge: both attempts crash; there is nothing to judge, so every item
    has no answer (0, no error) naming both."""
    records, model = _walk("solver_judge", {"solve_a", "solve_b"}, tmp_path, monkeypatch)
    _no_answer(records, {"solve_a", "solve_b"})
    assert model.calls("judge") == 0


def test_verify_with_no_draft_has_no_answer(tmp_path, monkeypatch):
    """[walker] solver_verify: the draft crashes; read_draft cannot run, and the item has
    no answer naming solve_draft and its cause -- not read_draft's FileNotFoundError."""
    records, model = _walk("solver_verify", {"solve_draft"}, tmp_path, monkeypatch)
    _no_answer(records, {"solve_draft"})
    assert all(r["missing_steps"]["solve_draft"]["cause"] == "output_limit" for r in records)
    assert "read_draft" not in records[0]["fail_reason"]
    assert model.calls("verify") == 0


def test_skill_graph_decides_with_the_notes_it_has(tmp_path, monkeypatch):
    """[walker] skill_graph: precision crashes; decide gets compare's and hypotheses' notes
    and a no-notes line for precision, and every item is graded."""
    from refactory_packages.livemath.skill_graph import NO_NOTES

    records, model = _walk("skill_graph", {"precision"}, tmp_path, monkeypatch)
    _answered(records, {"precision"})
    assert model.calls("decide") == ITEMS
    assert all(NO_NOTES in prompt for prompt in model.prompts("decide"))


@pytest.mark.parametrize("crash", [{"precision", "hypotheses"}, {"compare"}])
def test_skill_graph_with_nothing_to_decide_from_has_no_answer(tmp_path, monkeypatch, crash):
    """[walker] skill_graph: both fork steps crash, or compare (which every later step
    needs) does; the item has no answer."""
    records, model = _walk("skill_graph", crash, tmp_path, monkeypatch)
    _no_answer(records, crash)
    assert model.calls("decide") == 0


# ── the spine ──────────────────────────────────────────────────────────────


def _on_spine(name: str, plan: dict[str, list[str]], tmp_path: Path) -> list[dict]:
    """Run the real mode on the spine drain with dispatched workers against a scripted
    dsh that answers by prompt (or system prompt) text."""
    dsh = Scripted(tmp_path / "dsh", plan)
    mode = _mode(name)
    state = mode.state_dict()
    run = tmp_path / "spine"

    def dispatched(workflow: Any) -> Any:
        return workers.lower_workflow(workflow, tmp_path / "specs", dsh.client().context(),
                                      lowering.node_declarations(state))

    run.mkdir()
    lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / "graphs", name="crash-livemath",
                                              transform=dispatched)
    env = {key: value for key, value in os.environ.items() if key != "REFACTORY_WORKER_CONTEXT"}
    env.update(REFACTORY_WORKDIR=str(run), PYTHONPATH=str(Path(__file__).resolve().parent),
               LIVEMATH_DATA_PATH=str(FIXTURE_DIR), PATH=f"{PYTHON_BIN}:{env.get('PATH', '')}")
    done = subprocess.run([str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path),
                           str(run), f"{PYTHON_BIN}/python", "120000"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300, env=env)
    assert done.returncode == 0, f"stdout: {done.stdout[-2000:]}\nstderr: {done.stderr[-3000:]}"
    return json.loads((run / "facts.json").read_text())["item_scores"]


@needs_spine
def test_on_the_spine_skill_graph_decides_with_the_notes_it_has(tmp_path):
    """[spine] skill_graph: precision (told apart by its system prompt) crashes every
    session; every item is graded, with precision missing."""
    records = _on_spine("skill_graph", {"characterization": [WORKER_CRASH],
                                        "": ["<answer>B</answer>"]}, tmp_path)
    _answered(records, {"precision"})


@needs_spine
def test_on_the_spine_judge_with_no_attempt_has_no_answer(tmp_path):
    records = _on_spine("solver_judge", {"Compare both attempts.": ["<answer>B</answer>"],
                                         "": [WORKER_CRASH]}, tmp_path)
    _no_answer(records, {"solve_a", "solve_b"})


@needs_spine
def test_on_the_spine_verify_with_no_draft_has_no_answer(tmp_path):
    records = _on_spine("solver_verify", {"Carefully verify the solver": ["<answer>B</answer>"],
                                          "": [WORKER_CRASH]}, tmp_path)
    _no_answer(records, {"solve_draft"})
    assert all(r["missing_steps"]["solve_draft"]["cause"] == "crash" for r in records)
