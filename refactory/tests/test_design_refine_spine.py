"""The light refine on the spine: every node its own process, and `refine` a knob.

On DSH a refine arrives as the `refine` knob, with the run's other knobs in
`REFACTORY_KNOBS`, and a node's task names only the node. So the light-route test runs
the lowered design graph's own commands through the node runner
(`python -m refactory_dsh.run`), one process each, with the environment the spine gives
them, against the fake GitHub of `test_delivery`. The agent nodes are stood in for:
their declared reads must be on disk, their prompt is rendered from the facts on disk as
the worker renders it, and their output is written as a worker would write it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import refactory_dsh
from refactory_dsh import lowering
from refactory_dsh.design import DesignMode
from refactory_dsh.design_ops import LIGHT_REFINE_STRATEGY_PLAN, REFINE_SCOPE_REASON
from refactory_dsh.nn.executor import RunContext, render
from refactory_dsh.run import declarations, main, update_facts
from refactory_dsh.steering import Ledger
from test_delivery import (  # noqa: F401  (_stage, _opened_golden and opened_pull are the fake GitHub's fixtures)
    NAME,
    _describe,
    _opened_golden,
    _stage,
    opened_pull,
)
from tests_support import DESIGN_DOC_ANSWER

#: Where the node processes import refactory_dsh from: this checkout.
SOURCE = str(Path(refactory_dsh.__file__).resolve().parents[1])

DESCRIPTION = "WHY-THIS-PULL-REQUEST: the tool prints one, and the next step adds a second tool."
BRIEF = "Add more.py that prints two."


#: Design's prepare checks the model credential (dm-257): the test's own home has a placeholder key.
pytestmark = pytest.mark.usefixtures("configured_model_home")


class SpineRun:
    """One design run's nodes, run as the spine runs them."""

    def __init__(self, run: Path, state: dict) -> None:
        self.run, self.state = run, state
        self.workflow = lowering.to_workflow(state)
        self.graph = DesignMode().build()
        self.env = lowering.knob_environment(state)
        lowering.write_declarations(state, run)

    def facts(self) -> dict:
        return json.loads((self.run / "facts.json").read_text())

    def missing(self, node_id: str) -> list[str]:
        """The node's declared reads that are not on disk."""
        return [read for read in declarations(self.run, node_id)["reads"] if not (self.run / read).exists()]

    def node(self, node_id: str) -> str:
        """Run a deterministic node's lowered command in its own process; its last line (a gate's verdict)."""
        node = self.workflow.nodes[node_id]
        python, *runner = (getattr(node, "evaluator_command", None) or node.command).split()
        assert (python, runner[:2]) == ("python", ["-m", "refactory_dsh.run"]), runner
        assert not self.missing(node_id), (node_id, self.missing(node_id))
        env = {**os.environ, **self.env, "PYTHONPATH": SOURCE, "PYTHONDONTWRITEBYTECODE": "1",
               "REFACTORY_WORKDIR": str(self.run),
               "DSH_WORKFLOW_NODE": json.dumps({"id": node_id, "type": type(node).__name__})}
        done = subprocess.run([sys.executable, *runner], env=env, cwd=self.run, capture_output=True,
                              text=True, timeout=600)
        assert done.returncode == 0, (node_id, done.stderr[-3000:])
        lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def agent(self, node_id: str, writes: dict[str, str], facts: dict[str, str]) -> str:
        """Stand in for an agent node; the prompt the worker would have rendered for it."""
        assert not self.missing(node_id), (node_id, self.missing(node_id))
        context = RunContext(self.run, knobs=json.loads(self.env[lowering.KNOBS_ENV]), facts=self.facts())
        prompt = render(self.graph.nodes[node_id].prompt_template, context)
        for name, text in writes.items():
            (self.run / name).write_text(text)
        update_facts(self.run, facts)
        return prompt


def _refine(url: str, tmp_path: Path, **knobs: str) -> SpineRun:
    """A design run whose knobs ask for a refine of ``url``, as the spine is given them."""
    state = DesignMode().state_dict()
    given = {"name": NAME, "brief": BRIEF, "check": "python tool.py", "refine": url, **knobs}
    for knob, value in given.items():
        state["knobs"][knob]["value"] = value
    run = tmp_path / "refine"
    run.mkdir()
    Ledger.seed(run, BRIEF)
    return SpineRun(run, state)


def test_the_spine_takes_the_light_route_when_refine_is_a_knob(opened_pull, tmp_path):
    """Real prepare and route_refine_design, as separate processes with `refine` a knob,
    take the light route: the design's reads are all there, before and after a person's
    reloop, its prompt shows the pull request both times, and the approval card names
    the refinement as the run's scale."""
    _describe(opened_pull.github, "acme/app", DESCRIPTION)
    spine = _refine(opened_pull.url, tmp_path)

    spine.node("prepare")
    assert (spine.facts()["research_scope"], spine.facts()["research_scope_reason"]) == (
        "scoped", REFINE_SCOPE_REASON)
    assert spine.node("route_refine_design") == "light"
    first = spine.agent("design_doc", {"design.md": DESIGN_DOC_ANSWER}, {"design": DESIGN_DOC_ANSWER})
    spine.node("check_design_claims")
    assert spine.node("gate_design_review") == "proceed"
    assert f"Scale: scoped — {REFINE_SCOPE_REASON}" in spine.facts()["design_review_status"]
    assert not spine.missing("gate_strategy")

    # The person reloops at gate_strategy: the plan is redone, then the design.
    synthesize = spine.agent("synthesize_strategy", {"strategy.json": '{"strategy": "REVISED-PLAN"}\n'},
                             {"strategy": "REVISED-PLAN"})
    second = spine.agent("design_doc", {"design.md": DESIGN_DOC_ANSWER}, {"design": DESIGN_DOC_ANSWER})
    spine.node("check_design_claims")
    assert spine.node("gate_design_review") == "proceed"
    assert not spine.missing("gate_strategy")
    assert "## The plan\nREVISED-PLAN\n" in second
    for prompt in (first, synthesize, second):
        assert not re.findall(r"\{\{\s*[A-Za-z_]\w*\s*\}\}", prompt)
        assert LIGHT_REFINE_STRATEGY_PLAN in prompt
        assert DESCRIPTION in prompt
        assert "tool.py" in prompt.split("### Changed files\n", 1)[1].splitlines()


def _run_node_main(run, target, node_id, node_type, monkeypatch):
    import io

    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node_id, "type": node_type}))
    err, out = io.StringIO(), io.StringIO()
    old_err, old_out = sys.stderr, sys.stdout
    sys.stderr, sys.stdout = err, out
    try:
        code = main([target, "--node", node_id])
    finally:
        sys.stderr, sys.stdout = old_err, old_out
    return code, out.getvalue(), err.getvalue()


def test_spine_route_refine_design_proceed_on_non_refine(tmp_path, monkeypatch):
    """route_refine_design on a non-refine run returns proceed and writes nothing."""
    mode = DesignMode()
    state = mode.state_dict()
    run = tmp_path / "run"
    lowering.write_declarations(state, run)

    # Write facts: a non-refine run.
    (run / "facts.json").write_text(json.dumps({
        "research_scope_reason": "",
    }))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({}))

    target = "refactory_dsh.design_ops:route_refine_design"
    code, stdout, stderr = _run_node_main(run, target, "route_refine_design", "gate", monkeypatch)

    assert code == 0, f"exit {code}: {stderr}"
    lines = [l.strip() for l in stdout.strip().splitlines() if l.strip()]
    assert lines[-1] == "proceed", f"expected verdict 'proceed', got lines: {lines}"

    # No sentinel files were written (the downstream nodes write them).
    assert not (run / "research-plan.json").exists()
    assert not (run / "pitfalls.json").exists()
    assert not (run / "strategy.json").exists()
