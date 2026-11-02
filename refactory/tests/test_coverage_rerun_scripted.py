"""Scripted walker run for the QA coverage re-run path.

A walker run where gate_qa finds an unverified requirement shows the
qa_director_rerun prompt dispatched with that requirement id, its review
merged by synthesize_qa, and the run proceeding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ── model that simulates an unverified requirement on the first QA round ──

class _CoverageRerunModel:
    """FakeModel subclass that makes the first intent-audit.json leave one
    requirement 'unverified' so gate_qa triggers a coverage_rerun, then
    marks it 'pass' on the second QA judge call."""

    def __init__(self, answers: list[str]) -> None:
        from tests_support import FakeModel
        self._inner = FakeModel(answers)
        self._qa_judge_calls = 0

    # Delegate everything
    @property
    def prompts(self):
        return self._inner.prompts

    @property
    def workdirs(self):
        return self._inner.workdirs

    @property
    def cwds(self):
        return self._inner.cwds

    @property
    def envs(self):
        return self._inner.envs

    def with_tool_policy(self, policy):
        self._inner.with_tool_policy(policy)
        return self

    def complete(self, prompt: str, *, system: str = "", workdir: object = None,
                 cwd: object = None, env: object = None, confine: bool = False) -> str:
        from tests_support import QA_JUDGE_MARK
        answer = self._inner.complete(prompt, system=system, workdir=workdir,
                                      cwd=cwd, env=env, confine=confine)
        # After the QA judge writes intent-audit.json, modify the first call's
        # audit to leave one requirement "unverified".
        if QA_JUDGE_MARK in prompt and workdir is not None:
            self._qa_judge_calls += 1
            audit_path = Path(str(workdir)) / "intent-audit.json"
            if audit_path.is_file() and self._qa_judge_calls == 1:
                audit = json.loads(audit_path.read_text())
                reqs = audit.get("requirements", [])
                if reqs:
                    # Make the first requirement "unverified"
                    reqs[0]["status"] = "unverified"
                    audit_path.write_text(json.dumps(audit))
        return answer


def _rerun_answers(emitted: str) -> list[str]:
    from tests_support import (RESEARCH_ANSWER, PITFALLS_ANSWER,
                               DESIGN_DOC_ANSWER, VERIFICATION_ANSWER)
    rerun_review = (
        "## ask-1 coverage probe\n\n"
        "ran: python tool.py\nresult: built\n\n"
        "All requirements now covered.\n"
    )
    return [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER, emitted,
        # reviews (health, code, qa) plus verifier
        "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER,
        # qa_director_rerun answer — becomes reviews/qa-rerun.md
        rerun_review,
        # overwatch (consumed by auto-judge as VERDICT: PASS → OVERWATCH_PASS)
        "VERDICT: PASS",
    ]


@pytest.mark.timeout(120)
def test_scripted_coverage_rerun_walker_run(tmp_path, monkeypatch):
    """A scripted walker run where gate_qa finds an unverified requirement
    shows the qa_director_rerun prompt dispatched with that requirement id,
    its review is what synthesize_qa merges, and the run proceeds."""
    from refactory_dsh.design import DesignMode
    from tests_support import git_repo, render

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path / "project"))
    repo = git_repo(tmp_path / "repo")
    emitted = render({"tool.py": "def build():\n    print('built')\n"})
    model = _CoverageRerunModel(_rerun_answers(emitted))
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    run = mode.run(
        tmp_path / "run", model=model,
        verdicts={"gate_strategy": "proceed"},
        task={"name": "cov_rerun", "brief": "a tool that prints built",
              "check": f"{sys.executable} tool.py",
              "target": str(repo)})

    assert run.exit_ok, run.error

    # 1. The qa_director_rerun prompt was dispatched with the requirement id.
    from tests_support import QA_JUDGE_MARK
    rerun_prompts = [p for p in model.prompts
                     if "coverage re-run" in p and "qa_coverage_rerun_ids" not in p
                     and "reviews/qa-rerun.md" in p]
    assert len(rerun_prompts) >= 1, (
        "qa_director_rerun prompt not found; prompts: "
        + str([p[:80] for p in model.prompts])
    )
    # The prompt should contain an unverified requirement id.
    rerun_prompt = rerun_prompts[0]
    assert "ask-" in rerun_prompt or "unverified" in rerun_prompt.lower(), (
        "qa_director_rerun prompt does not mention the unverified requirement"
    )

    # 2. The QA judge was called at least twice (initial + after re-run).
    qa_judge_prompts = [p for p in model.prompts if QA_JUDGE_MARK in p]
    assert len(qa_judge_prompts) >= 2, (
        f"Expected at least 2 QA judge calls, got {len(qa_judge_prompts)}"
    )

    # 3. synthesize_qa merged the rerun review.
    synthesis = (tmp_path / "run" / "reviews/qa-synthesized.md").read_text()
    assert "coverage re-run" in synthesis.lower() or "qa-rerun" in synthesis.lower(), (
        "synthesize_qa did not merge the coverage re-run review"
    )

    # 4. The run proceeded (exit_ok above) with no builder reloop from coverage.
    gate_text = (tmp_path / "run" / "reviews/qa-gate.txt").read_text()
    assert "proceed" in gate_text

    # 5. No more than one QA director rerun per round.
    rerun_count = sum(1 for p in model.prompts
                      if "coverage re-run" in p and "reviews/qa-rerun.md" in p)
    assert rerun_count == 1, f"Expected exactly 1 qa_director_rerun call, got {rerun_count}"


@pytest.mark.timeout(120)
def test_coverage_rerun_lowering_snapshot_contains_rerun_node():
    """The lowering snapshot for the spine includes qa_director_rerun and the
    updated edge, so both runtimes see the same graph structure."""
    snapshot_path = (Path(__file__).resolve().parent / "lowering_snapshots"
                     / "default_pack-design.graph.json")
    assert snapshot_path.exists(), f"snapshot missing: {snapshot_path}"
    graph = json.loads(snapshot_path.read_text())
    assert "qa_director_rerun" in graph["nodes"], "qa_director_rerun missing from snapshot"
    node = graph["nodes"]["qa_director_rerun"]
    assert node.get("_type") == "AgentNode"
    # The edge from gate_qa should target qa_director_rerun, not synthesize_qa
    coverage_edges = [e for e in graph["edges"]
                      if e["source"] == "gate_qa"
                      and e.get("condition") == "coverage_rerun"]
    assert len(coverage_edges) == 1
    assert coverage_edges[0]["target"] == "qa_director_rerun"
    # qa_director_rerun → synthesize_qa edge exists
    rerun_to_synth = [e for e in graph["edges"]
                      if e["source"] == "qa_director_rerun"
                      and e["target"] == "synthesize_qa"]
    assert len(rerun_to_synth) == 1
