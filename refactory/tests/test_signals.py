"""The optimizer-facing signals: priors, refusal findings, and convergence.

The signals issue. None of this makes the search correct; all of it makes it better,
which is why it follows the correctness work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refactory_dsh import nn, registry
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate, Proposal, RandomOptimizer
from refactory_dsh.optim.base import ProposalContext

MODE = "circle_packing/adaptive"


def _config(root: Path, **overrides: object) -> LoopConfig:
    base: dict[str, object] = dict(
        epochs=2, batch_size=3, optimizer="random", seed=5, runs_root=root
    )
    base.update(overrides)
    return LoopConfig(**base)  # type: ignore[arg-type]


def _ctx(mode: nn.Mode | None = None) -> ProposalContext:
    mode = mode or registry.resolve(Path("."), MODE)[1]()
    described = mode.describe()
    return ProposalContext(
        knobs=[{"name": name, **spec} for name, spec in described["knobs"].items()],
        nodes=described["nodes"],
        edges=described["edges"],
        catalogue=nn.edits.catalogue(),
        insertable=[node.to_state() for node in mode.insertable()],
        guidance=described["guidance"],
        preferred_edits=described["preferred_edits"],
    )


# ── author priors ──────────────────────────────────────────────────────────
def test_the_mode_states_what_is_worth_doing():
    """A knob's description says what it is; guidance says what to do with it."""
    mode = registry.resolve(Path("."), MODE)[1]()
    described = mode.describe()
    assert described["guidance"], "the running example offers no guidance"
    assert "deterministic" in described["guidance"]
    assert described["preferred_edits"] == ["set_knob", "insert_after"]


def test_the_prompt_carries_the_guidance_and_the_preferred_operations():
    prompt = _ctx().render()
    assert "## What the author says is worth doing" in prompt
    assert "deterministic" in prompt
    assert "`set_knob`" in prompt


def test_a_mode_with_no_guidance_says_so_rather_than_showing_nothing():
    class Silent(nn.Mode):
        name = "silent"

        def build(self) -> nn.Graph:
            graph = nn.Graph()
            graph.add(nn.Node("only", "op", call="loop_ops:write_best",
                              writes=("best.json",), produces=("best",)))
            return graph

    described = Silent().describe()
    assert described["guidance"] == ""
    prompt = _ctx(Silent()).render()
    assert "offered no guidance" in prompt


# ── item results ───────────────────────────────────────────────────────────
_CAPPED = ("solve: RuntimeError: model completion failed (exit 1): dsh: max-tokens: "
           "the model reached its output token limit before it finished (32768 tokens)")


def _records() -> list[dict]:
    return [
        {"id": "m2_0", "passed": True, "score": 1.0, "error_text": None},
        {"id": "m2_1", "passed": False, "score": 0.0, "error_text": None,
         "fail_reason": "Qh7+ is not mate: the king escapes to f8"},
        {"id": "m2_2", "passed": False, "score": 0.0, "error_text": _CAPPED, "fail_reason": _CAPPED},
        {"id": "m2_3", "passed": False, "score": 0.0, "error_text": _CAPPED, "fail_reason": _CAPPED},
        {"id": "m2_4", "passed": False, "score": 0.0,
         "error_text": "solve: CompletionTimeout: model completion timed out after 600s"},
    ]


def test_the_prompt_tells_errored_items_from_wrong_answers():
    """The loop handed the proposer its parent's item records, and the prompt never
    showed them: a run that lost a third of its items to the output token limit read
    exactly like one that answered them wrong."""
    ctx = _ctx()
    ctx.item_records = _records()
    prompt = ctx.render()
    assert "## Item results of the candidate being edited" in prompt
    assert "5 train item(s): 1 passed, 1 answered wrong, 3 failed with an error, mean score 0.2" in prompt
    # Errors are grouped by cause, the most common first, with example items.
    errors = prompt.split("Errors, by cause:")[1]
    assert errors.index("max-tokens") < errors.index("timed out")
    assert "| 2 | " in errors and "m2_2, m2_3" in errors
    # A wrong answer shows the grader's reason; an errored item is not listed twice.
    wrong = prompt.split("Wrong answers, with the grader's reason:")[1]
    assert "the king escapes to f8" in wrong
    assert "m2_2" not in wrong


def test_no_item_records_adds_no_section():
    assert "Item results" not in _ctx().render()


def test_the_item_section_is_bounded():
    from refactory_dsh.optim.base import ITEM_EVIDENCE_CHARS, ITEM_EVIDENCE_ROWS

    ctx = _ctx()
    ctx.item_records = [
        {"id": f"i{k}", "passed": False, "score": 0.0, "error_text": f"cause {k}: " + "x" * 1000}
        for k in range(ITEM_EVIDENCE_ROWS + 4)
    ]
    section = ctx.render().split("## Item results of the candidate being edited")[1].split("\n## ")[0]
    assert "(4 more cause(s) not shown)" in section
    assert "x" * ITEM_EVIDENCE_CHARS not in section


def test_the_sampler_favours_the_preferred_operations(tmp_path):
    """A prior, not a restriction: the sampler still covers the catalogue."""
    mode = registry.resolve(Path("."), MODE)[1]()
    parent = Candidate.from_state(mode.state_dict())
    ctx = _ctx(mode)
    ops: list[str] = []
    for seed in range(30):
        candidates, _rejections = RandomOptimizer(seed=seed).propose_valid(
            ctx, 2, parents=[parent], invariants=mode.invariants()
        )
        for candidate in candidates:
            ops.extend(edit.get("op") for edit in candidate.edits[-1:])
    assert "set_knob" in ops, "the preferred operation was never proposed"
    assert len(set(ops)) > 1, "the sampler only proposed what the author suggested"


# ── the block signal ───────────────────────────────────────────────────────
def test_an_advisory_block_declares_that_it_cannot_change_the_score():
    mode = registry.resolve(Path("."), MODE)[1]()
    blocks = {node.id: node for node in mode.insertable()}
    assert blocks["critic"].score_effect == "none"
    assert blocks["restart_on_stall"].score_effect == "affects"
    prompt = _ctx(mode).render()
    assert "score_effect" in prompt and "cannot change the score" in prompt


def test_an_unknown_score_effect_is_refused():
    with pytest.raises(nn.GraphError, match="unknown score_effect"):
        nn.Node("x", "op", call="loop_ops:write_best", score_effect="maybe")


def test_score_effect_travels_in_the_state():
    mode = registry.resolve(Path("."), MODE)[1]()
    state = mode.state_dict()
    assert isinstance(state, dict)
    from refactory_dsh.nn import Graph

    graph = Graph.from_state(state["graph"])
    assert graph.nodes["exit"].score_effect == "affects"


# ── refusal findings ───────────────────────────────────────────────────────
def test_a_dominant_refusal_is_reported_as_a_finding_about_the_catalogue(tmp_path):
    from refactory_dsh.optim import Optimizer

    class AlwaysBad(Optimizer):
        """Proposes an operation that is not in the catalogue, every time."""

        name = "always-bad"

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return [
                Proposal(edit={"op": "levitate", "target": "eval"}, parent=parents[0].id)
            ] * n

    loop = Loop(
        MODE, _config(tmp_path / "run", epochs=1, batch_size=1),
        project=Path("."), optimizer=AlwaysBad(k=2),
    )
    report = loop.run()
    dominant = report["dominant_refusal"]
    assert dominant["code"], "no refusal was recorded"
    assert dominant["share"] >= 0.5
    assert dominant["finding"], "a dominant refusal must be stated as a finding"
    assert dominant["code"] == "E-op", dominant
    assert "not about the optimizer" in dominant["finding"]
    markdown = (tmp_path / "run" / "report.md").read_text()
    assert "## Why proposals were refused" in markdown
    assert dominant["finding"][:40] in markdown


def test_no_dominant_refusal_when_nothing_was_refused(tmp_path):
    """A clean run should not invent a finding."""
    # Use the knob_only preset so the random optimizer only picks knob edits
    # that succeed, avoiding structural E-shape refusals.  F14 removed
    # set_node_field/replace_node which shifted the old seed's refusal rate.
    loop = Loop(
        MODE,
        _config(tmp_path / "run", optimizer="random",
                edit_preset="knob_only"),
        project=Path("."),
    )
    loop.run()
    assert loop._dominant_refusal()["code"] is None or loop._dominant_refusal()["finding"] == ""


# ── convergence ────────────────────────────────────────────────────────────
def test_running_out_of_ideas_is_reported_as_convergence_not_a_refusal(tmp_path):
    """A no-op means the optimizer is out of ideas within the surface it can see."""
    from refactory_dsh.optim import Optimizer

    class Stuck(Optimizer):
        name = "stuck"

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            currently = parents[0].state["knobs"]["order"]["value"]
            return [
                Proposal(edit={"op": "set_knob", "name": "order", "value": currently},
                         parent=parents[0].id)
            ] * n

    loop = Loop(MODE, _config(tmp_path / "run", epochs=3), project=Path("."), optimizer=Stuck(k=3))
    report = loop.run()
    convergence = report["convergence"]
    assert convergence["reason"] == "surface_exhausted", convergence
    assert "widen" in convergence["detail"] or "widening" in convergence["detail"]
    markdown = (tmp_path / "run" / "report.md").read_text()
    assert "## Did the search run out of ideas?" in markdown


def test_a_productive_run_reports_no_convergence(tmp_path):
    report = Loop(MODE, _config(tmp_path / "run"), project=Path(".")).run()
    assert report["convergence"]["reason"] is None


# ── efficiency, and adaptive k ─────────────────────────────────────────────
def test_the_report_says_how_many_attempts_a_candidate_cost(tmp_path):
    report = Loop(MODE, _config(tmp_path / "run"), project=Path(".")).run()
    efficiency = report["efficiency"]
    assert efficiency["batches"] >= 1
    assert efficiency["effective_k"], "the report must say which k was used"
    assert efficiency["why"], "and why that k"
    if efficiency["candidates"]:
        assert efficiency["attempts_per_candidate"] is not None


def test_k_rises_when_a_batch_is_refused_more_than_it_is_accepted(tmp_path):
    """The signal that the optimizer is close to something legal, not out of ideas."""
    from refactory_dsh.optim import Optimizer

    class HalfBad(Optimizer):
        name = "half-bad"
        calls = 0

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            HalfBad.calls += 1
            out = []
            for index in range(n):
                if index % 2:
                    out.append(Proposal(
                        edit={"op": "remove_node", "id": "exit"}, parent=parents[0].id
                    ))
                else:
                    currently = parents[0].state["knobs"]["grid"]["value"]
                    values = [v for v in (6, 10, 18, 24) if v != currently]
                    out.append(Proposal(
                        edit={"op": "set_knob", "name": "grid",
                              "value": values[HalfBad.calls % len(values)]},
                        parent=parents[0].id,
                    ))
            return out

    loop = Loop(
        MODE, _config(tmp_path / "run", epochs=4, batch_size=2, attempts=1),
        project=Path("."), optimizer=HalfBad(k=1),
    )
    report = loop.run()
    effective = report["efficiency"]["effective_k"]
    assert max(effective) > 1, (
        f"k never adapted past the configured 1 despite a refusal-heavy batch: {effective}"
    )
