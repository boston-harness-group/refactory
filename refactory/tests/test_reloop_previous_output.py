"""A step a loop's judge sends back sees its own previous output with the judge's feedback.

The judge's reason is about the work the step did on its previous visit, so the step's
next prompt carries both: the feedback, then the step's own last output
(`agent_delivery.previous_output`), ahead of its prompt. Both runtimes build that prompt
in `agent_delivery.with_feedback` -- the walker from its gate's reason, the spine's worker
(`worker_node`) from the feedback file the spine wrote -- so the text is the same. These
tests run the loop op on S1 (livemath/solver: one agent ending its pass) and S4
(livemath/skill_graph), including ``back_to``, on the walker, and the spine's worker on
the same inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

import structure_fuzz as sf
from refactory_dsh import agent_delivery
from refactory_dsh.optim import Candidate
from test_code_nodes import _Model

FIRST = "FIRST-ATTEMPT-TEXT <answer>A</answer>"
SECOND = "SECOND-ATTEMPT-TEXT <answer>B</answer>"
REASON = "JUDGE-REASON: option A overstates the result"


def _solver_seed() -> sf.Seed:
    from refactory_packages.livemath.mode import Solver

    return sf.Seed("livemath_solver", Solver, livemath=True)


def _seeds() -> dict[str, sf.Seed]:
    return {"S1": _solver_seed(), "S4": sf.seeds()["livemath_skill_graph"]}


def _looped(seed: sf.Seed, edit: dict) -> dict:
    applier = sf.Applier(seed)
    parent = Candidate.from_state(applier.seed_state)
    out = applier.attempt(parent, edit, {parent.id})
    assert isinstance(out, Candidate), out
    return out.state


def _judge(state: dict, step: str) -> str:
    gate = next(n for n in state["graph"]["nodes"].values()
                if n["kind"] == "gate" and n.get("id", "").startswith(step))
    reads = gate.get("required_reads") or gate.get("reads")
    return next(n["id"] for n in state["graph"]["nodes"].values()
                if n["kind"] == "agent" and reads and reads[0] in (n.get("writes") or [])[:1])


def _walk(seed: sf.Seed, state: dict, tmp_path: Path, sent_back: str, judge: str) -> _Model:
    model = _Model(sf._answers(sf._graph_of(state)),
                   turns={sent_back: [FIRST, SECOND], judge: [f"reloop: {REASON}", "proceed"]})
    result = sf.run_walker(seed, state, tmp_path / "walker", model)
    assert result.exit_ok, result.error
    return model


def _prompts(model: _Model, node: str) -> list[str]:
    return [prompt for who, prompt, _system in model.log if who == node]


CASES = {
    # S1: the loop around the one agent that ends the pass.
    "S1 loop": ("S1", {"op": "loop", "step": "solve", "judge": "agent", "max_iterations": 1},
                "solve", "solve"),
    # S4: a loop around a step with a next step.
    "S4 loop": ("S4", {"op": "loop", "step": "decide", "judge": "agent", "max_iterations": 1},
                "decide", "decide"),
    # S4: a judge after a later step sends the work back to an earlier agent.
    "S4 back_to": ("S4", {"op": "loop", "step": "notes_prompts", "back_to": "compare",
                          "judge": "agent", "max_iterations": 1}, "compare", "notes_prompts"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_walker_a_step_sent_back_sees_its_last_output_and_the_judges_feedback(case, tmp_path):
    name, edit, sent_back, judged = CASES[case]
    seed = _seeds()[name]
    state = _looped(seed, edit)
    model = _walk(seed, state, tmp_path, sent_back, _judge(state, judged))
    prompts = _prompts(model, sent_back)
    first_visit = prompts[0]
    assert agent_delivery.FEEDBACK_HEADING not in first_visit and FIRST not in first_visit
    again = next(p for p in prompts[1:] if REASON in p)
    head = again.split(agent_delivery.PREVIOUS_HEADING, 1)
    assert len(head) == 2, again[:400]
    assert head[0].startswith(agent_delivery.FEEDBACK_HEADING) and REASON in head[0]
    # Its own output from its previous visit, then its prompt as before.
    assert FIRST in head[1] and SECOND not in head[1]
    assert again.endswith(first_visit), "the prompt itself follows, unchanged"


def test_the_spine_worker_builds_the_same_prompt_from_the_spine_s_feedback_file(tmp_path):
    """Spine: the worker (`worker_node.execute`) reads the feedback file the spine wrote
    for the step sent back and the step's own output still in its run directory, and
    sends the prompt the walker sends on that visit, byte for byte."""
    seed = _seeds()["S1"]
    state = _looped(seed, CASES["S1 loop"][1])
    model = _walk(seed, state, tmp_path, "solve", _judge(state, "solve"))
    first_visit, walker_again = _prompts(model, "solve")[0], next(
        p for p in _prompts(model, "solve")[1:] if REASON in p)

    workdir = tmp_path / "spine"
    prompts = workdir / ".refactory" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "solve.md").write_text(first_visit)
    (prompts / "solve.feedback.md").write_text(REASON)
    (workdir / "answer.txt").write_text(FIRST)
    spec = {"schema_version": 1, "node_id": "solve", "prompt": "fallback", "reads": [],
            "writes": ["answer.txt"], "tool_policy": "none",
            "model_context": {"provider": "test", "model": "test", "worker_backend": "dsh"}}
    sent: list[str] = []

    class Client:
        def __init__(self, **_kwargs): pass
        def for_node(self, node, role="worker"): return self
        def with_tool_policy(self, policy): return self
        def context(self): return {"provider": "test", "model": "test"}
        def route(self): return {"worker_backend": "dsh"}

    def begin(_workdir, node, prompt, writes, **_kwargs):
        sent.append(prompt)
        return {"prompt": prompt, "node": node, "status": "pending"}

    manifest = {"id": "solve", "type": "AgentNode", "prompt_file": str(prompts / "solve.md"),
                "feedback_file": str(prompts / "solve.feedback.md")}
    from refactory_dsh.run import NODE_ENV

    with mock.patch.dict(os.environ, {NODE_ENV: json.dumps(manifest)}), \
            mock.patch("refactory_dsh.model.DshModelClient", Client), \
            mock.patch("refactory_dsh.agent_delivery.begin", begin), \
            mock.patch("refactory_dsh.agent_delivery.finish",
                       lambda *a, **k: {"status": "delivered"}), \
            mock.patch("refactory_dsh.run._knobs", return_value={}), \
            mock.patch("refactory_dsh.run._read_facts", return_value={}), \
            mock.patch("refactory_dsh.nn.executor.render", side_effect=lambda p, c: p), \
            mock.patch("refactory_dsh.nn.executor.RunContext"):
        from refactory_dsh import worker_node

        try:
            worker_node.execute(spec, workdir)
        except Exception:  # noqa: BLE001 - only the prompt it sent is under test
            pass
    assert sent and sent[0] == walker_again


def test_a_step_that_reads_its_run_directory_with_tools_is_not_handed_its_output(tmp_path):
    (tmp_path / "out.txt").write_text(FIRST)
    assert agent_delivery.previous_output(tmp_path, ["out.txt"], sees_run_dir=True) == ""
    assert agent_delivery.previous_output(tmp_path, ["out.txt"], sees_run_dir=False) == FIRST
    assert agent_delivery.previous_output(tmp_path, ["missing.txt"], sees_run_dir=False) == ""
    # No feedback, nothing added; a fence longer than any run of backticks in the output.
    assert agent_delivery.with_feedback("P", "", FIRST) == "P"
    said = agent_delivery.with_feedback("P", REASON, "x ```` y")
    assert "`````\nx ```` y\n`````" in said


def test_a_long_previous_output_is_cut_to_its_head_and_tail_and_the_cut_said(tmp_path):
    cap = agent_delivery.PREVIOUS_CHARS
    text = "H" * cap + "MIDDLE" + "T" * cap
    (tmp_path / "out.txt").write_text(text)
    said = agent_delivery.previous_output(tmp_path, ["out.txt"], sees_run_dir=False)
    assert said.startswith("H" * (cap // 2) + "\n[... ") and said.endswith("\n" + "T" * (cap // 2))
    assert f"[... {len(text) - cap} characters omitted ...]" in said and "MIDDLE" not in said
    assert len(said) < cap + 100
