"""One design pass: research, strategy and design go straight to the person's gate.

On runs b39a, b39b, b40a and b41a the design phase took 81-87 minutes: design_doc was
written three or four times, sent back by the claims check for a quote a few lines from
its citation, a quote with an ellipsis or a bare file name, and by the design-review
agent's BLOCKING findings. Now there is no reviewer: design_doc runs once unless the
design has no ``## Changes``, the claims check resolves drift and lists what it cannot
confirm, and the research and pitfalls gates never fail the run. These tests drive the
real design graph with scripted answers, and the gate through the node runner the spine
uses.
"""
from __future__ import annotations

import io
import json
import re
import sys

import pytest

from refactory_dsh import create_ops, design_contract, design_ops, nn
from refactory_dsh.design import DesignMode
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER,
                           SCOPED_RESEARCH_ANSWER, FakeModel, git)

DESIGN_WRITER = "Write the design document for this artifact"
RESEARCH_DIRECTOR = "You are the research director."
PITFALLS_RESEARCHER = "You are the pitfalls researcher."

#: pkg/sub/engine.py: run_engine is defined on line 34 and returns on line 36.
ENGINE = ("".join(f"# filler {n}\n" for n in range(1, 34))
          + "def run_engine(config, *, retries=3) -> int:\n"
          + "    total = len(config)\n"
          + "    return total + retries  # final count\n")


def _target(root):
    """A checkout with a package two levels down, so a bare name must be resolved."""
    files = {"app.py": "print('app')\n",
             "pkg/__init__.py": "",
             "pkg/sub/__init__.py": "",
             "pkg/sub/engine.py": ENGINE,
             "pkg/sub/helpers.py": "import os\n\ndef helper():\n    return os.sep\n"}
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def _with_claims(*claims: str, design: str = DESIGN_DOC_ANSWER) -> str:
    return design + "\n## Existing code\n\n" + "".join(f"- {claim}\n" for claim in claims)


#: Three claims the old check sent back, each true of the checkout.
DRIFTED = "`run_engine()` at `pkg/sub/engine.py:30`: `def run_engine(config, *, retries=3) -> int:`"
ELLIPSIS = "`pkg/sub/engine.py:36`: `return total … # final count`"
BARE_NAME = "`helper()` at `helpers.py:3`: `def helper():`"


def _run_to_the_gate(tmp_path, answers):
    """Run the design graph until the approval gate, which has no verdict here."""
    repo = _target(tmp_path / "repo")
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    run = mode.run(tmp_path / "run", model=model,
                   task={"name": "pass_tool", "brief": "a tool",
                         "check": f"{sys.executable} -c pass", "target": str(repo)})
    assert run.gate_stop == "gate_strategy", run.error
    return run, model


FULL_HEAD = [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan"]


def _card(run) -> list[str]:
    return run.facts["design_review_status"].splitlines()


def _writes(model) -> list[str]:
    return [p for p in model.prompts if DESIGN_WRITER in p]


@pytest.mark.parametrize("research, head", [
    (RESEARCH_ANSWER, ["research", "pitfalls", "perspectives", "plan"]),
    (SCOPED_RESEARCH_ANSWER, ["research", "pitfalls", "plan"]),
], ids=["full", "scoped"])
def test_no_reviewer_runs_between_the_design_and_the_gate(tmp_path, research, head):
    answers = [research, PITFALLS_ANSWER, *(["perspectives"] if "perspectives" in head else []),
               "one plan", DESIGN_DOC_ANSWER]
    run, model = _run_to_the_gate(tmp_path, answers)

    # Every scripted answer was used, and the last model call was the design itself.
    assert len(model.prompts) == len(head) + 1
    assert DESIGN_WRITER in model.prompts[-1]
    assert run.facts["design_review_status"].startswith("Design review: PASS")
    # And in the graph: from design_doc to gate_strategy only deterministic nodes.
    graph = DesignMode().build()
    seen, frontier = set(), ["design_doc"]
    while frontier:
        node = frontier.pop()
        for edge in graph.out_edges(node):
            if edge.condition != "reloop" and edge.target not in seen | {"gate_strategy"}:
                seen.add(edge.target)
                frontier.append(edge.target)
    assert seen and all(graph.nodes[node].kind in ("op", "gate") for node in seen), seen


def test_drift_ellipsis_and_a_bare_name_cost_no_round(tmp_path):
    run, model = _run_to_the_gate(
        tmp_path, [*FULL_HEAD, _with_claims(DRIFTED, ELLIPSIS, BARE_NAME)])

    assert len(_writes(model)) == 1
    assert run.facts["claims_check"] == {
        "unverified": [], "notes": ["`pkg/sub/engine.py:30`: cited 30, found at 34"]}
    card = _card(run)
    assert card[0] == "Design review: PASS"
    assert card[card.index("Claims check (advisory): 1 note") + 1] == (
        "- `pkg/sub/engine.py:30`: cited 30, found at 34")
    assert not [line for line in card if "unverified" in line.lower()]


def test_an_unverified_claim_is_listed_first_and_costs_no_round(tmp_path):
    wrong = "`run_engine()` at `pkg/sub/engine.py:34`: `def run_engine(config):`"
    run, model = _run_to_the_gate(tmp_path, [*FULL_HEAD, _with_claims(wrong, DRIFTED)])

    assert len(_writes(model)) == 1
    card = _card(run)
    assert card[0] == "Design review: PASS" and card[1].startswith("Scale: full")
    # First after the headline and the scale line, ahead of the drift notes.
    assert card[2] == "Unverified claims (high, advisory): 1"
    assert card[3].startswith("- unverified claim: quote `def run_engine(config):` not within "
                              "20 lines of `pkg/sub/engine.py:34`")
    assert card[4] == "Claims check (advisory): 1 note"


def test_a_design_missing_changes_goes_back_exactly_once(tmp_path):
    without_changes = re.sub(r"## Changes\n\n.*?\n\n(?=## )", "", DESIGN_DOC_ANSWER, flags=re.S)
    without_changes = without_changes.replace("# Do the thing", "# pass_tool")
    assert "## Changes" not in without_changes
    run, model = _run_to_the_gate(tmp_path, [*FULL_HEAD, without_changes, without_changes])

    writes = _writes(model)
    assert len(writes) == 2
    assert "the design is missing the section the build needs, or it is empty: ## Changes" \
        in writes[1]
    # The rewrite is also shown what is advisory, to fix while it is there.
    assert "the design's title is the run name" in writes[1]
    card = _card(run)
    assert card[:2] == ["Design review: FAIL, 1 unresolved",
                        "- the design is missing the section the build needs, or it is "
                        "empty: ## Changes"]
    assert run.facts["design_review_attempts_used"] == 2


def test_a_rewrite_is_counted_not_a_visit(tmp_path):
    """A design that passed, sent back by the person at gate_strategy, that then has no
    ## Changes still gets its one rewrite."""
    ctx = nn.RunContext(tmp_path)
    ctx.write_text("design.md", DESIGN_DOC_ANSWER)
    assert design_ops.gate_design_review(ctx) == "proceed"
    ctx.write_text("design.md", DESIGN_DOC_ANSWER.replace("## Changes", "## Notes"))
    assert design_ops.gate_design_review(ctx) == "reloop"
    assert design_ops.gate_design_review(ctx) == "proceed"
    assert ctx.facts["design_review_rewrites"] == 1


@pytest.mark.parametrize("changes, verdict", [(True, "proceed"), (False, "reloop")])
def test_the_gate_runs_through_the_node_runner(tmp_path, monkeypatch, changes, verdict):
    """The spine runs gate_design_review as its own process (`refactory_dsh.run`): it
    reads the claims check's facts, writes the card, and exits 0 on either verdict."""
    from refactory_dsh import lowering
    from refactory_dsh.run import main

    run = tmp_path / "run"
    run.mkdir()
    lowering.write_declarations(DesignMode().state_dict(), run)
    design = DESIGN_DOC_ANSWER if changes else DESIGN_DOC_ANSWER.replace("## Changes", "## Notes")
    (run / "facts.json").write_text(json.dumps({
        "name": "pass_tool", "research_scope": "full",
        "research_scope_reason": "no usable footprint in the research",
        "claims_check": {"unverified": ["`gone.py` is not in the checkout"],
                         "notes": ["`a.py:3`: cited 3, found at 5"]}}))
    (run / "design.md").write_text(design)
    (run / "brief.json").write_text(json.dumps({"name": "pass_tool", "brief": "a tool"}))
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "gate_design_review", "type": "gate"}))
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    code = main(["refactory_dsh.design_ops:gate_design_review", "--node", "gate_design_review"])

    assert code == 0, err.getvalue()
    assert out.getvalue().strip().splitlines()[-1] == verdict
    facts = json.loads((run / "facts.json").read_text())
    if verdict == "proceed":
        assert facts["design_review_status"].splitlines()[:6] == [
            "Design review: PASS",
            "Scale: full — no usable footprint in the research",
            "Unverified claims (high, advisory): 1",
            "- unverified claim: `gone.py` is not in the checkout",
            "Claims check (advisory): 1 note",
            "- `a.py:3`: cited 3, found at 5"]
    else:
        assert facts["design_review_findings"] == "(no findings yet)"


# ── the light template's consumers ──────────────────────────────────────────────

LIGHT = ("# Make the tool print what it built\n\n"
         "## Summary\n\nIt prints nothing now.\n\n"
         "## Changes\n\n- `tool.py` — print the built name\n\n"
         "## Tests\n\n- `tests/test_tool.py::test_build` — the name is printed\n")


def test_a_light_design_is_read_as_a_contract_design():
    parsed = design_contract.parse(LIGHT, "pass_tool")
    assert parsed.has_contract
    assert parsed.title == "Make the tool print what it built"
    assert parsed.summary == "It prints nothing now."
    assert parsed.changes == ["`tool.py` — print the built name"]
    assert parsed.rule_sites == []
    # The pull request lists the diff's files, as for a design with no rule sites.
    assert design_contract.build_what_changed(parsed, ["b.py", "tool.py"]) == (
        ["b.py", "tool.py"], [])


def test_corrections_keep_a_light_design_and_restore_one_that_lost_its_changes(tmp_path):
    from refactory_dsh.delivery import _validate_design_after_corrections

    corrected = LIGHT + "\n## Corrections at approval\n\n- \"print the name\" — § Changes\n"
    (tmp_path / "design.md.pre_corrections").write_text(LIGHT)
    (tmp_path / "design.md").write_text(corrected)
    _validate_design_after_corrections(tmp_path)
    assert (tmp_path / "design.md").read_text() == corrected
    (tmp_path / "design.md").write_text(corrected.replace("## Changes", "## Notes"))
    _validate_design_after_corrections(tmp_path)
    assert (tmp_path / "design.md").read_text() == LIGHT


NO_SUMMARY = LIGHT.replace("## Summary\n\nIt prints nothing now.\n\n", "")
CORRECTION = "\n## Corrections at approval\n\n- \"print the name\" \u2014 \u00a7 Changes\n"


@pytest.mark.parametrize("before, after, restored", [
    # A section the design never had is not lost: the corrections stay.
    (NO_SUMMARY, NO_SUMMARY + CORRECTION, False),
    # A section it had and the corrections deleted, or emptied, is.
    (LIGHT, LIGHT.replace("## Summary", "## Notes") + CORRECTION, True),
    (LIGHT, LIGHT.replace("It prints nothing now.", "") + CORRECTION, True),
    (NO_SUMMARY, NO_SUMMARY.replace("## Changes", "## Notes") + CORRECTION, True),
], ids=["never-had-summary", "lost-summary", "emptied-summary", "no-summary-lost-changes"])
def test_corrections_restore_only_a_section_the_design_had(tmp_path, before, after, restored):
    from refactory_dsh.delivery import _validate_design_after_corrections

    (tmp_path / "design.md.pre_corrections").write_text(before)
    (tmp_path / "design.md").write_text(after)
    _validate_design_after_corrections(tmp_path)
    assert (tmp_path / "design.md").read_text() == (before if restored else after)


# ── research and pitfalls never fail a design run ───────────────────────────────

ONE_DIRECTION = json.dumps([{"focus": "where the tool prints", "slug": "printing",
                             "finding": "tool.py prints nothing", "ids": ["FND-printing-1"]}])


def test_research_with_one_direction_proceeds_with_a_card_note(tmp_path):
    run, model = _run_to_the_gate(
        tmp_path, [ONE_DIRECTION, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER])

    assert len([p for p in model.prompts if RESEARCH_DIRECTOR in p]) == 1
    assert ("- research: the plan has 1 direction(s); three to five are suggested"
            in _card(run))


def test_unparseable_research_goes_back_once_then_proceeds(tmp_path, monkeypatch):
    # The spine's worker refuses a declared input that is not on disk (research/ for the
    # plan and design steps); hold the walker run to the same rule.
    from pathlib import Path
    from refactory_dsh import agent_delivery

    deliver = agent_delivery.begin

    def strict_begin(workdir, node, prompt, writes, **kwargs):
        for name in kwargs.get("reads", []):
            assert (Path(workdir) / name).exists(), f"{node}: required worker input is missing: {name}"
        return deliver(workdir, node, prompt, writes, **kwargs)

    monkeypatch.setattr(agent_delivery, "begin", strict_begin)
    prose = "The subagents are still running in the background."
    run, model = _run_to_the_gate(
        tmp_path, [prose, prose, PITFALLS_ANSWER, "perspectives", "one plan", DESIGN_DOC_ANSWER])

    research = [p for p in model.prompts if RESEARCH_DIRECTOR in p]
    assert len(research) == 2
    assert "is not a JSON array of {focus, slug, finding} objects" in research[1]
    assert len(_writes(model)) == 1
    assert any(line.startswith("- research: research-plan.json is not a JSON array")
               and line.endswith("the design proceeded without a research plan")
               for line in _card(run))


def test_unparseable_pitfalls_go_back_once_then_proceed(tmp_path):
    run, model = _run_to_the_gate(
        tmp_path, [RESEARCH_ANSWER, "no pitfalls", "still none", "perspectives", "one plan",
                   DESIGN_DOC_ANSWER])

    assert len([p for p in model.prompts if PITFALLS_RESEARCHER in p]) == 2
    assert len(_writes(model)) == 1
    assert any(line.startswith("- pitfalls: pitfalls.json holds no parseable pitfalls")
               for line in _card(run))


def test_pitfalls_with_bad_entries_proceed_with_the_problems_listed(tmp_path):
    ctx = nn.RunContext(tmp_path)
    ctx.write_text("pitfalls.json", json.dumps([{"id": "PIT-1", "risk": "severe"}, "text"]))
    assert design_ops.gate_pitfalls(ctx) == "proceed"
    advisory = ctx.facts["pitfalls_advisory"]
    assert "id 'PIT-1' is not in PIT-NNN form" in advisory
    assert "entry 1: not a JSON object" in advisory


def test_create_research_gate_is_unchanged(tmp_path):
    from refactory_dsh.create import CreateMode

    assert CreateMode().build().nodes["gate_research"].evaluator == (
        "refactory_dsh.create_ops:gate_research")
    assert DesignMode().build().nodes["gate_research"].evaluator == (
        "refactory_dsh.design_ops:gate_design_research")
    ctx = nn.RunContext(tmp_path, knobs={"research_attempts": 2})
    ctx.write_text("research-plan.json", ONE_DIRECTION)
    ctx.write_text("research/printing.md", "### FND-printing-1 — tool.py prints nothing\n")
    verdict = create_ops.gate_research(ctx)
    assert verdict == "reloop"
    assert "the plan has 1 direction(s); three to five are required" in verdict.reason
    with pytest.raises(ValueError, match="the research is still not usable after 2 attempt"):
        create_ops.gate_research(ctx)


def test_only_design_modes_builder_checks_each_correction_fails_without_its_change():
    """Every extra build round the QA judge caused was a gate correction no test exercised,
    or a fix whose test still passed with the fix reverted (approved 2026-10-26). Design
    mode's builder is told to run each test with its change reverted in its workspace (it
    works in a clone of the attempt, so it reverts there, not in a draft emit.txt); no other
    mode is."""
    from refactory_dsh.design import EMIT_CHECKLIST, WORKSPACE_BRIEFING
    from test_lowering_smoke import _mode_classes

    emit = DesignMode().build().nodes["emit"].prompt_template
    assert EMIT_CHECKLIST in emit and emit.index(WORKSPACE_BRIEFING) < emit.index(EMIT_CHECKLIST)
    for line in ("each correction made at the design gate and each ask in the brief",
                 "revert that change in the workspace and run the same test again: it must fail",
                 "A correction counts as met only if a test fails without it",
                 "calls the helper under test", "a verdict list nothing consumes",
                 "results the test built by hand"):
        assert line in EMIT_CHECKLIST, line
    others = {key: cls().build().nodes.get("emit") for key, cls in _mode_classes() if cls is not DesignMode}
    assert "default_pack-design_fast" in others
    assert [key for key, node in others.items()
            if node is not None and "## Before you stop" in node.prompt_template] == []


def test_the_workspace_builder_runs_the_tests_its_change_touches_not_the_full_suite():
    """In the live trial (dm-149) the workspace builder ran the full suites several times:
    11.6 of its 27 minutes, before the self-check and the check ran them again. Design's and
    design_fast's builders are told to run the tests their change touches and the brief's
    own example, and that the check runs the full suite after they answer (approved
    2026-10-27); design's still reverts each change and reruns its test. No other mode's
    builder is told this."""
    from refactory_dsh.design import EMIT_CHECKLIST, WORKSPACE_BRIEFING
    from test_lowering_smoke import _mode_classes

    modes = dict(_mode_classes())
    told = {key: cls().build().nodes.get("emit") for key, cls in modes.items()}
    assert sorted(key for key, node in told.items()
                  if node is not None and WORKSPACE_BRIEFING in node.prompt_template) == [
        "default_pack-design", "default_pack-design_fast"]
    for key in ("default_pack-design", "default_pack-design_fast"):
        emit = told[key].prompt_template
        for line in ("Run there, as often as you need, the tests your change touches",
                     "the test files you changed or added",
                     "the tests that import the modules you changed",
                     "Run the brief's own example or CLI path there too",
                     ("Before you stop, run `refactory-targeted-tests` and fix what fails. It "
                      "runs exactly what the self-check will run for your change. "
                      "{{builder_tests}}"),
                     ("Do not run the whole test suite: the full check runs after you finish, "
                      "beside the reviews."),
                     ("If you wrap a command in `timeout`, give it about twice the time stated "
                      "above")):
            assert line in emit, (key, line)
        # The only mention of the whole suite is the one telling the builder not to run it,
        # and no fixed timeout is suggested: the time stated above is the guide.
        assert emit.count("whole test suite") == 1, key
        assert "full suite" not in emit and "5m" not in emit and "timeout -k" not in emit, key
        assert "check command runs there" not in emit, key
    assert "revert that change in the workspace and run the same test again: it must fail" in (
        EMIT_CHECKLIST)
    assert EMIT_CHECKLIST in told["default_pack-design"].prompt_template
