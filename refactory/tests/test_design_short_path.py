"""A scoped change takes the short design path: no perspectives step.

`route_strategy` sends a scoped change from the pitfalls straight to the plan; a full
design weighs perspectives first. At either scale the design goes from the claims check
to the design-review gate and on to the person: there is no design reviewer. The routing
gate answers `scoped` or `proceed`, the edge labels both runtimes follow; `proceed` is
also what a person's skip of a gate sends on the spine.
"""
from __future__ import annotations

import json
import sys

from refactory_dsh import lowering
from refactory_dsh.design import DesignMode
from refactory_dsh.design_ops import SCOPED_STRATEGY_PLAN, gate_design_review, route_strategy
from test_design_contract_extensions import FakeCtx, _make_design
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
                           RESEARCH_ANSWER, SCOPED_RESEARCH_ANSWER, VERIFICATION_ANSWER,
                           FakeModel, git_repo, render)

STRATEGY_DIRECTOR = "You are the strategy director."
PLAN_STEP = "Compile the perspectives into one plan"


def _edges(graph):
    return {(e.source, e.target, e.condition) for e in graph.edges}


def test_the_design_graph_routes_on_the_scale():
    graph = DesignMode().build()
    edges = _edges(graph)
    # route_refine_design sits between prepare and research, which the sizer now
    # precedes (moved on purpose: the sizing chain starts on its proceed edge, at
    # route_sizing, which sends a refine straight to research).
    assert ("prepare", "route_refine_design", None) in edges
    assert ("route_refine_design", "route_sizing", "proceed") in edges
    assert ("route_refine_design", "design_doc", "light") in edges
    assert ("gate_research", "classify_scale", "proceed") in edges
    # Moved on purpose: the widening gate sits between classify_scale and the pitfalls.
    assert ("classify_scale", "widen_research", None) in edges
    assert ("widen_research", "pitfalls_researcher", "proceed") in edges
    assert ("gate_pitfalls", "route_strategy", "proceed") in edges
    assert ("route_strategy", "strategy_director", "proceed") in edges
    assert ("route_strategy", "synthesize_strategy", "scoped") in edges
    # The same design path at every scale: no reviewer between the design and the person.
    assert ("check_design_claims", "gate_design_review", None) in edges
    assert ("gate_design_review", "gate_strategy", "proceed") in edges
    assert "design_review" not in graph.nodes and "route_design_review" not in graph.nodes
    # No other way into the step a scoped change skips.
    assert {e for e in edges if e[1] == "strategy_director"} == {
        ("route_strategy", "strategy_director", "proceed")}


def test_the_spine_gets_the_same_routes():
    workflow = lowering.to_workflow(DesignMode().state_dict())
    node = workflow.nodes["route_strategy"]
    assert node.evaluator_type == "fn"
    assert node.evaluator_command.endswith("refactory_dsh.design_ops:route_strategy")
    refine_node = workflow.nodes["route_refine_design"]
    assert refine_node.evaluator_type == "fn"
    assert refine_node.evaluator_command.endswith("refactory_dsh.design_ops:route_refine_design")
    lowered = {(e.source, e.target, e.condition) for e in workflow.edges}
    assert ("prepare", "route_refine_design", None) in lowered
    assert ("route_refine_design", "route_sizing", "proceed") in lowered
    sizing = workflow.nodes["route_sizing"]
    assert sizing.evaluator_type == "fn"
    assert sizing.evaluator_command.endswith("refactory_dsh.design_ops:route_sizing")
    assert ("route_refine_design", "design_doc", "light") in lowered
    assert ("route_strategy", "synthesize_strategy", "scoped") in lowered
    assert ("check_design_claims", "gate_design_review", None) in lowered
    assert "design_review" not in workflow.nodes and "route_design_review" not in workflow.nodes


def test_the_routing_gate_answers_by_the_scale(tmp_path):
    from refactory_dsh.nn.executor import RunContext

    scoped = RunContext(tmp_path / "scoped", knobs={}, facts={"research_scope": "scoped"}, task={})
    assert route_strategy(scoped) == "scoped"
    assert scoped.facts["strategy_plan"] == SCOPED_STRATEGY_PLAN
    # The skipped step's outputs exist, empty, for the steps that declare them as inputs.
    assert json.loads((tmp_path / "scoped" / "strategy-plan.json").read_text()) == []
    assert (tmp_path / "scoped" / "strategy").is_dir()
    full = RunContext(tmp_path / "full", knobs={}, facts={"research_scope": "full"}, task={})
    assert route_strategy(full) == "proceed"
    assert "strategy_plan" not in full.facts
    assert not (tmp_path / "full" / "strategy-plan.json").exists()


_SCOPED_REASON = "touches 1 existing function(s) (app.main at app.py:1), adds nothing new, no shared engine"
SCOPED = {"name": "short", "has_pitfalls_step": False, "research_scope": "scoped",
          "research_scope_reason": _SCOPED_REASON}


def test_the_gate_reads_no_review():
    # A review left from anywhere is not read: nothing reviews the design.
    review = "BLOCKING: the design is wrong\n\n## Notes\n- a stale note\n\nVERDICT: FAIL"
    ctx = FakeCtx(files={"reviews/design-review.md": review, "design.md": _make_design(),
                         "brief.json": json.dumps({"brief": "a tool"})}, facts=dict(SCOPED))
    assert gate_design_review(ctx) == "proceed"
    assert ctx.facts["design_review_status"] == (
        "Design review: PASS\nScale: scoped \u2014 " + _SCOPED_REASON)
    # Its own check still sends back a design missing ## Changes, once, and only then
    # does it reach the person with the problem shown.
    ctx = FakeCtx(files={"design.md": _make_design().replace("## Changes", "## Notes"),
                         "brief.json": json.dumps({"brief": "a tool"})}, facts=dict(SCOPED))
    first = gate_design_review(ctx)
    assert first == "reloop" and "## Changes" in first.reason
    assert gate_design_review(ctx) == "proceed"
    assert ctx.facts["design_review_status"].startswith("Design review: FAIL, 1 unresolved")


def _design_run(tmp_path, brief, answers):
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task={"name": "short_tool", "brief": brief,
                         "check": f"{sys.executable} tool.py", "target": str(repo)})
    return run, model


BUILD = [render({"tool.py": "def build():\n    print('built')\n"}),
         "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]


def test_a_scoped_change_runs_the_short_path_end_to_end(tmp_path, monkeypatch):
    # The spine's worker refuses a declared input that is not on disk; hold the walker
    # run to the same rule, so a step the short path skips cannot leave one missing.
    from pathlib import Path
    from refactory_dsh import agent_delivery

    deliver = agent_delivery.begin

    def strict_begin(workdir, node, prompt, writes, **kwargs):
        for name in kwargs.get("reads", []):
            assert (Path(workdir) / name).exists(), f"{node}: required worker input is missing: {name}"
        return deliver(workdir, node, prompt, writes, **kwargs)

    monkeypatch.setattr(agent_delivery, "begin", strict_begin)
    run, model = _design_run(tmp_path, "Make tool.py print built.", [
        SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *BUILD])

    assert run.exit_ok, run.error
    assert run.facts["research_scope"] == "scoped"
    assert "research_scope_reason" in run.facts
    assert not [p for p in model.prompts if STRATEGY_DIRECTOR in p]
    plan = next(p for p in model.prompts if PLAN_STEP in p)
    assert "no perspectives: this is a scoped change" in plan
    assert run.facts["design_review_status"].startswith(
        "Design review: PASS\nScale: scoped \u2014 ")


def test_a_scoped_design_the_checks_block_is_rewritten_once_before_the_gate(tmp_path):
    """Both 2026-10-22 measurement runs reached the person with a blocking finding the
    design writer could have fixed; a scoped design gets that one rewrite, now only for
    the section the build needs (a missing Ask mapping is not even checked)."""
    import re
    assert "## Changes" in DESIGN_DOC_ANSWER
    without_changes = re.sub(r"## Changes\n\n.*?\n\n(?=## )", "", DESIGN_DOC_ANSWER, flags=re.S)
    assert "## Changes" not in without_changes
    run, model = _design_run(tmp_path, "Make tool.py print built.", [
        SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", without_changes, DESIGN_DOC_ANSWER,
        *BUILD])

    assert run.exit_ok, run.error
    writes = [p for p in model.prompts if "Write the design document for this artifact" in p]
    assert len(writes) == 2
    assert "the design is missing the section the build needs, or it is empty: ## Changes" \
        in writes[1]
    assert run.facts["design_review_status"].startswith("Design review: PASS")


def test_a_full_design_keeps_both_steps(tmp_path):
    run, model = _design_run(tmp_path, "a tool", [
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER,
        *BUILD])

    assert run.exit_ok, run.error
    assert run.facts["research_scope"] == "full"
    assert len([p for p in model.prompts if STRATEGY_DIRECTOR in p]) == 1
    assert run.facts["design_review_status"].startswith(
        "Design review: PASS\nScale: full \u2014 ")


# ── the sizing chain before research, and the widening gate before the pitfalls ──

SIZING_CHAIN = {
    ("route_refine_design", "route_sizing", "proceed"),
    ("route_sizing", "clear_sizing", "proceed"),
    ("route_sizing", "research_director", "refine"),
    ("clear_sizing", "fork_size", None),
    ("fork_size", "join_size", None),
    ("join_size", "presize_scale", None),
    ("presize_scale", "research_director", None),
    ("classify_scale", "widen_research", None),
    ("widen_research", "research_director", "reloop"),
    ("widen_research", "pitfalls_researcher", "proceed"),
}


def test_the_sizing_chain_sits_before_research():
    graph = DesignMode().build()
    workflow = lowering.to_workflow(DesignMode().state_dict())
    lowered = {(e.source, e.target, e.condition) for e in workflow.edges}
    for edges in (_edges(graph), lowered):
        assert SIZING_CHAIN <= edges
        # The light route, the research gate and the strategy route are as they were.
        assert ("route_refine_design", "design_doc", "light") in edges
        assert {e for e in edges if e[0] == "route_refine_design"} == {
            ("route_refine_design", "route_sizing", "proceed"),
            ("route_refine_design", "design_doc", "light")}
        # A refine keeps its scale: route_sizing sends it to research past the sizer.
        assert {e for e in edges if e[0] == "route_sizing"} == {
            ("route_sizing", "clear_sizing", "proceed"),
            ("route_sizing", "research_director", "refine")}
        assert {e for e in edges if e[0] == "gate_research"} == {
            ("gate_research", "research_director", "reloop"),
            ("gate_research", "classify_scale", "proceed")}
        assert {e for e in edges if e[0] == "route_strategy"} == {
            ("route_strategy", "strategy_director", "proceed"),
            ("route_strategy", "synthesize_strategy", "scoped")}
        # The sizer is reached through its fork, not an edge.
        assert not {e for e in edges if "size_change" in e[:2]}
    assert workflow.nodes["fork_size"].targets == ["size_change"]
    assert workflow.nodes["fork_size"].optional_targets == ["size_change"]
    assert workflow.nodes["join_size"].sources == ["size_change"]


def test_every_design_path_passes_pitfalls():
    """Every path from the route's proceed edge to the approval gate passes widen_research and
    then pitfalls_researcher: no route skips the pitfalls."""
    graph = DesignMode().build()

    def walk(node, seen, passed):
        if node == "gate_strategy":
            assert passed == ["widen_research", "pitfalls_researcher"], seen
            return 1
        targets = [e.target for e in graph.out_edges(node)]
        if graph.nodes[node].kind == "fork":
            targets = [graph.fork_join(node)]
        count = 0
        for target in targets:
            if target in seen:  # a reloop: the path goes on from the earlier visit
                continue
            step = passed + [target] if target in ("widen_research", "pitfalls_researcher") \
                and target not in passed else passed
            count += walk(target, seen | {target}, step)
        return count

    # From route_sizing: both its sized and its refine route.
    assert walk("route_sizing", {"route_refine_design", "route_sizing"}, []) > 1
