"""Tests for the pitfalls researcher and its mechanical enforcement.

Each test fails without the corresponding production change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh import nn, pitfalls as pitfalls_mod
from refactory_dsh.design import DesignMode
from refactory_dsh.nn import Verdict
from tests_support import (
    DESIGN_DOC_ANSWER,
    FakeModel,
    PITFALLS_ANSWER,
    RESEARCH_ANSWER,
    REVIEW_ANSWERS,
    VERIFICATION_ANSWER,
    render,
)


# ── validation ─────────────────────────────────────────────────────────────

#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def test_validate_pitfalls_accepts_valid():
    raw = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "pinning_tests",
        "evidence": "tests/test_thing.py:10",
        "prescription": "keep this assertion",
    }])
    assert pitfalls_mod.validate_pitfalls(raw) == []


def test_validate_pitfalls_catches_invalid_json():
    assert pitfalls_mod.validate_pitfalls("not json")[0].startswith("pitfalls.json is not valid JSON")


def test_validate_pitfalls_catches_empty_array():
    assert "non-empty" in pitfalls_mod.validate_pitfalls("[]")[0]


def test_validate_pitfalls_catches_bad_id():
    raw = json.dumps([{
        "id": "BAD", "risk": "high", "category": "pinning_tests",
        "evidence": "f.py:1", "prescription": "fix it",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw)
    assert any("PIT-NNN" in p for p in problems)


def test_validate_pitfalls_catches_bad_risk():
    raw = json.dumps([{
        "id": "PIT-001", "risk": "critical", "category": "pinning_tests",
        "evidence": "f.py:1", "prescription": "fix it",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw)
    assert any("risk" in p for p in problems)


def test_validate_pitfalls_catches_bad_category():
    raw = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "unknown_cat",
        "evidence": "f.py:1", "prescription": "fix it",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw)
    assert any("category" in p for p in problems)


def test_validate_pitfalls_catches_empty_prescription():
    raw = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "pinning_tests",
        "evidence": "f.py:1", "prescription": "",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw)
    assert any("prescription" in p for p in problems)


# ── coverage ───────────────────────────────────────────────────────────────

def _pit(pid: str = "PIT-001", risk: str = "high") -> dict:
    return {"id": pid, "risk": risk, "category": "pinning_tests",
            "evidence": "f.py:1", "prescription": "keep it"}


def _contract(body: str = "", anti: str = "") -> str:
    base = (
        "# Do the thing\n\n"
        "## Summary\n\nFix it.\n\n"
        "## Failure trace\n\nN/A — this is a new feature.\n\n"
        "## Ask mapping\n\n"
        "| Ask (quoted from brief) | Design section | Test |\n"
        "|---|---|---|\n"
        '| "do a thing" | § Changes | test.py |\n\n'
        "## Rule and sites\n\n"
        "- `file.py:1` — **[changed]**\n\n"
        "## Docs\n\n"
        "- docs/models.md\n\n"
        "## Changes\n\n"
        "- file.py\n"
    )
    if body:
        base += f"\n{body}\n"
    if anti:
        base += f"\n## Anti-patterns to avoid\n\n{anti}\n"
    return base


# ── where the design cites each pitfall (dm-262: quoted, never judged) ─────

def _citations(design: str, *pits: dict) -> dict[str, list]:
    return {entry["id"]: entry["quotes"]
            for entry in pitfalls_mod.pitfall_citations(design, list(pits) or [_pit()])}


def test_a_pitfall_the_design_never_cites_is_not_mentioned():
    assert _citations(_contract()) == {"PIT-001": []}
    entry = pitfalls_mod.pitfall_citations(_contract(), [_pit()])[0]
    assert pitfalls_mod.citation_line(entry) == "PIT-001 (high): not mentioned"


def test_every_line_citing_a_pitfall_is_quoted_under_its_heading_at_most_two():
    design = _contract(body="### Notes\n\nPIT-001 is why the base is read twice.",
                       anti="- **PIT-001** (high): addressed at `file.py:1`")
    # The Anti-patterns entry first, where a design answers pitfalls; then design order.
    assert _citations(design) == {"PIT-001": [
        ("Anti-patterns to avoid", "PIT-001 (high): addressed at `file.py:1`"),
        ("Notes", "PIT-001 is why the base is read twice.")]}
    entry = pitfalls_mod.pitfall_citations(design, [_pit()])[0]
    assert pitfalls_mod.citation_line(entry) == (
        "PIT-001 (high): \u00a7 Anti-patterns to avoid: \"PIT-001 (high): addressed at `file.py:1`\"; "
        "\u00a7 Notes: \"PIT-001 is why the base is read twice.\"")
    three = _contract(body="PIT-001 first.\n\nPIT-001 second.", anti="- PIT-001 third.")
    assert [text for _, text in _citations(three)["PIT-001"]] == ["PIT-001 third.", "PIT-001 first."]
    # A subsection of Anti-patterns is still where the design answers pitfalls.
    nested = _contract(body="PIT-001 first.", anti="### Details\n\nPIT-001 under a subsection.")
    assert _citations(nested)["PIT-001"] == [("Details", "PIT-001 under a subsection."),
                                             ("Changes", "PIT-001 first.")]


def test_a_list_item_is_quoted_with_the_lines_that_continue_it():
    design = _contract(anti="- **PIT-001**\n  The base is read once, so the cache key holds.\n"
                            "- PIT-002 is another\n\nPIT-003 later.")
    assert _citations(design, _pit())["PIT-001"] == [
        ("Anti-patterns to avoid", "PIT-001 The base is read once, so the cache key holds.")]


def test_a_hash_line_in_a_code_fence_is_not_a_heading():
    design = _contract(body="```python\n# PIT-001: read the base once\nbase = read()\n```\n\n"
                            "PIT-001 is settled by the line above.")
    assert _citations(design) == {"PIT-001": [
        ("Changes", "PIT-001: read the base once"),
        ("Changes", "PIT-001 is settled by the line above.")]}


def test_a_summary_mention_or_a_set_aside_is_quoted_like_any_other():
    """Nothing is decided from the words: the person reads them."""
    summary = _contract().replace("Fix it.", "PIT-001 is important.")
    assert _citations(summary) == {"PIT-001": [("Summary", "PIT-001 is important.")]}
    aside = _contract(anti="- PIT-001: **[set-aside: out of scope]**")
    assert _citations(aside) == {"PIT-001": [("Anti-patterns to avoid",
                                              "PIT-001: [set-aside: out of scope]")]}


def test_an_id_is_cited_only_as_itself_and_a_long_line_is_cut_around_it():
    long = "word " * 60 + "PIT-001 is settled by reading the base once " + "more " * 40
    design = _contract(anti="- PIT-0010 is another pitfall\n- " + long)
    quotes = _citations(design)["PIT-001"]
    assert len(quotes) == 1
    text = quotes[0][1]
    assert text.startswith("\u2026") and text.endswith("\u2026")
    assert "PIT-001 is settled by reading the base once" in text
    assert len(text) <= pitfalls_mod.QUOTE_LIMIT


# ── gate_design_review: the card quotes, the advisory names what is never cited ──

def _review(tmp_path, design: str, pits: list) -> nn.RunContext:
    import refactory_dsh.design_ops as ops

    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "has_pitfalls_step": True})
    ctx.write_text("design.md", design)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    ctx.write_json("pitfalls.json", pits)
    assert ops.gate_design_review(ctx) == "proceed"
    return ctx


def test_the_gate_lists_quotes_and_names_high_risk_pitfalls_never_mentioned(tmp_path):
    ctx = _review(tmp_path, _contract(anti="- PIT-002: see \u00a7 Changes"),
                  [_pit(), _pit("PIT-002", "low"), _pit("PIT-003", "medium")])
    assert ctx.facts["pitfall_coverage"] == (
        "Pitfall citations:\n"
        "  PIT-001 (high): not mentioned\n"
        "  PIT-002 (low): \u00a7 Anti-patterns to avoid: \"PIT-002: see \u00a7 Changes\"\n"
        "  PIT-003 (medium): not mentioned")
    # Only the high-risk one is advised on; nothing says addressed or missing.
    assert [line for line in ctx.facts["design_review_advisory"].splitlines()
            if "pitfall" in line] == ["- high-risk pitfall(s) the design does not mention: PIT-001"]
    for verdict in ("addressed", "MISSING", "not checked"):
        assert verdict not in ctx.facts["pitfall_coverage"] + ctx.facts["design_card"]


def test_a_design_without_an_anti_patterns_section_is_quoted_all_the_same(tmp_path):
    ctx = _review(tmp_path, _contract(body="Reads the base once, which settles PIT-001."), [_pit()])
    assert ctx.facts["pitfall_coverage"] == (
        "Pitfall citations:\n  PIT-001 (high): \u00a7 Changes: "
        "\"Reads the base once, which settles PIT-001.\"")
    assert "pitfall" not in ctx.facts["design_review_advisory"]


def test_gate_design_review_skips_without_pitfalls_step(tmp_path):
    """A graph without a research stage records why it has no pitfalls."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({
        "source": str(tmp_path),
        "design_review_attempts_used": 0,
        # has_pitfalls_step not set → skip
    })
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", _contract())
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed"
    coverage = ctx.facts.get("pitfall_coverage", "")
    assert "skipped" in coverage.lower() or "no pitfalls" in coverage.lower()


# ── design graph structure ─────────────────────────────────────────────────

def test_design_graph_has_pitfalls_nodes():
    graph = DesignMode().build()
    assert "pitfalls_researcher" in graph.nodes
    assert "gate_pitfalls" in graph.nodes
    # Wiring: gate_research → classify_scale → widen_research → pitfalls_researcher →
    # gate_pitfalls → route_strategy, which goes on to strategy_director for a full design.
    # (widen_research moved in on purpose: a lean research is widened before the pitfalls.)
    proceed_edges = graph.out_edges("gate_research", condition="proceed")
    assert len(proceed_edges) == 1
    assert proceed_edges[0].target == "classify_scale"
    cs_edges = graph.out_edges("classify_scale")
    assert cs_edges[0].target == "widen_research"
    assert [e.target for e in graph.out_edges("widen_research", condition="proceed")] == [
        "pitfalls_researcher"]
    pit_edges = graph.out_edges("pitfalls_researcher")
    assert pit_edges[0].target == "gate_pitfalls"
    gp_proceed = graph.out_edges("gate_pitfalls", condition="proceed")
    assert gp_proceed[0].target == "route_strategy"
    assert graph.out_edges("route_strategy", condition="proceed")[0].target == "strategy_director"
    gp_reloop = graph.out_edges("gate_pitfalls", condition="reloop")
    assert gp_reloop[0].target == "pitfalls_researcher"


def test_design_prompts_contain_pitfalls():
    """The strategy and design prompts contain {{pitfalls}}."""
    graph = DesignMode().build()
    for node_id in ("strategy_director", "synthesize_strategy", "design_doc",
                    "code_reviewer"):
        assert "{{pitfalls}}" in graph.nodes[node_id].prompt_template, (
            f"{node_id} prompt must contain {{{{pitfalls}}}}"
        )


def test_gate_pitfalls_reloops_once_when_nothing_parses(tmp_path):
    """gate_pitfalls sends the step back when pitfalls.json holds no parseable pitfalls."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "pitfalls_attempts_used": 0})
    ctx.write_text("pitfalls.json", "not json")
    import refactory_dsh.design_ops as ops
    verdict = ops.gate_pitfalls(ctx)
    assert verdict == "reloop"
    assert "no parseable pitfalls" in verdict.reason


def test_gate_pitfalls_never_stops_the_run(tmp_path):
    """Once its one resend is spent, gate_pitfalls proceeds with the problem listed."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "pitfalls_attempts_used": 1})
    ctx.write_text("pitfalls.json", "not json")
    import refactory_dsh.design_ops as ops
    assert ops.gate_pitfalls(ctx) == "proceed"
    assert "the design proceeded without pitfalls" in ctx.facts["pitfalls_advisory"]


def test_gate_pitfalls_proceeds_on_valid(tmp_path):
    """gate_pitfalls proceeds when pitfalls.json is valid."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "pitfalls_attempts_used": 0})
    ctx.write_text("pitfalls.json", PITFALLS_ANSWER)
    import refactory_dsh.design_ops as ops
    assert ops.gate_pitfalls(ctx) == "proceed"


# ── fixture repository with pinning test ───────────────────────────────────

def test_fixture_repo_gets_pitfall_for_pinning_test(tmp_path):
    """A fixture repo whose test pins behaviour gets a pitfall naming that test.

    Creates a repository with a test file containing ``assert timeout == 300``,
    writes a brief that asks to change the timeout, scripts a pitfall answer
    referencing the test file, line and assertion, and validates it through
    gate_pitfalls — matching what a scripted model would produce, as the
    existing research tests do.

    Fails without the pitfalls_researcher node and gate_pitfalls gate:
    there would be no node to write pitfalls.json and no gate to validate it.
    """
    # ── fixture repository with a pinning test ──
    repo = tmp_path / "repo"
    repo.mkdir()
    test_dir = repo / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_config.py"
    test_file.write_text(
        "def test_default_timeout():\n"
        "    timeout = 300\n"
        "    assert timeout == 300\n"
    )

    # ── scripted pitfall answer referencing the pinning test ──
    pitfall_answer = json.dumps([{
        "id": "PIT-001",
        "risk": "high",
        "category": "pinning_tests",
        "evidence": "tests/test_config.py:3",
        "prescription": (
            "The test asserts `timeout == 300`. The change must keep this "
            "assertion or deliberately update it with a reason."
        ),
    }])

    # ── run context pointing at the fixture repo as its source ──
    run = tmp_path / "run"
    ctx = nn.RunContext(run)
    ctx.facts.update({"source": str(repo), "pitfalls_attempts_used": 0})
    ctx.write_text("pitfalls.json", pitfall_answer)

    # gate_pitfalls validates the schema and the evidence file:line.
    import refactory_dsh.design_ops as ops
    assert ops.gate_pitfalls(ctx) == "proceed"

    # Verify the written pitfalls.json has the required fields.
    written = json.loads(ctx.read_text("pitfalls.json"))
    assert len(written) == 1
    p = written[0]
    assert p["category"] == "pinning_tests"
    assert "tests/test_config.py" in p["evidence"]
    assert ":3" in p["evidence"]
    assert pitfalls_mod.PIT_ID_RE.match(p["id"])
    assert p["risk"] == "high"
    assert p["prescription"]

    # The evidence file actually exists in the fixture repo.
    assert (repo / "tests" / "test_config.py").is_file()
    content = (repo / "tests" / "test_config.py").read_text()
    assert "assert timeout == 300" in content


# ── three-form evidence classifier ────────────────────────────────────────

def test_new_project_design_run_passes_pitfalls_gate(tmp_path):
    """T0: A scripted new-project design run whose pitfalls use absent: and
    brief.json evidence gets past gate_pitfalls and completes successfully.

    Red without C1: brief.json evidence (no colon) is rejected by the old
    colon check, gate_pitfalls reloops, and the run dies with ValueError.
    """
    new_project_pitfalls = json.dumps([
        {
            "id": "PIT-001",
            "risk": "high",
            "category": "scope_boundaries",
            "evidence": "absent: pyproject.toml",
            "prescription": "create a pyproject.toml",
        },
        {
            "id": "PIT-002",
            "risk": "medium",
            "category": "pinning_tests",
            "evidence": "brief.json",
            "prescription": "the brief asks for tests but none exist yet",
        },
    ])
    design_doc = (
        "# Do the thing\n\n"
        "## Summary\n\nFix it properly.\n\n"
        "## Failure trace\n\nN/A \u2014 this is a new feature.\n\n"
        "## Ask mapping\n\n"
        "| Ask (quoted from brief) | Design section | Test |\n"
        "|---|---|---|\n"
        '| "a tool" | \u00a7 Changes | tool.py |\n\n'
        "## Rule and sites\n\n"
        "- `tool.py:1` \u2014 **[changed]**\n\n"
        "## Docs\n\n"
        "- docs/models.md\n\n"
        "## Changes\n\n"
        "- tool.py\n\n"
        "## Research and perspective findings\n\n"
        "- FND-api-surface-1: addressed at `tool.py:1`\n"
        "- FND-existing-modes-1: addressed at \u00a7 Changes\n"
        "- FND-acceptance-1: addressed at `tool.py:1`\n"
        "- FND-simplest-1: addressed at `tool.py:1`\n"
        "- FND-criteria-1: addressed at \u00a7 Changes\n\n"
        "## Anti-patterns to avoid\n\n"
        "- PIT-001: addressed at `tool.py:1`\n"
        "- PIT-002: [set-aside: no tests exist yet in a new project]\n"
    )
    answers = [
        RESEARCH_ANSWER,
        new_project_pitfalls,
        "perspectives",
        "one plan",
        design_doc,
        render({"tool.py": "print('built')\n"}),
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]
    result = DesignMode().run(
        tmp_path / "run",
        model=FakeModel(answers),
        verdicts={"gate_strategy": "proceed"},
        task={"name": "my_tool", "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok, result.error


def test_empty_target_absent_evidence_passes_gate(tmp_path):
    """T1 (pin): absent: evidence with a truly absent path passes gate_pitfalls.

    This is a pin test — the colon in 'absent: pyproject.toml' means the old
    code did not reject it either. The red-green proofs are T0, T2, T2b, T4.
    """
    source = tmp_path / "repo"
    source.mkdir()
    run = tmp_path / "run"
    ctx = nn.RunContext(run)
    ctx.facts.update({"source": str(source), "pitfalls_attempts_used": 0})
    pitfalls_data = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "absent: pyproject.toml",
        "prescription": "create a pyproject.toml",
    }])
    ctx.write_text("pitfalls.json", pitfalls_data)
    import refactory_dsh.design_ops as ops
    assert ops.gate_pitfalls(ctx) == "proceed"


def test_false_absence_claim_refused_naming_path(tmp_path):
    """T2: A false absence claim (the path exists) is refused, naming the path.

    Red without C1: the old code passes 'absent: pyproject.toml' because it
    contains a colon, then the regex finds no file:N match and silently skips.
    """
    source = tmp_path / "repo"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\n")
    raw = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "absent: pyproject.toml",
        "prescription": "create a pyproject.toml",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw, str(source))
    assert any("false absence" in p and "pyproject.toml" in p for p in problems)


def test_false_glob_absence_claim_refused_naming_matches(tmp_path):
    """T2b: A false glob absence claim is refused, naming match(es).

    Red without C1: the old code treats 'absent: tests/test_*.py' as a
    potential file:line citation, finds no digit after *, and silently passes.
    """
    source = tmp_path / "repo"
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test_foo.py").write_text("")
    (source / "tests" / "test_bar.py").write_text("")
    raw = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "absent: tests/test_*.py",
        "prescription": "no tests yet",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw, str(source))
    assert any("false absence" in p and "test_" in p for p in problems)


def test_no_citation_still_refused():
    """T3 (pin): Evidence with no colon, no absent:, no run-file pattern is
    still refused.

    Green both before and after: the existing colon check already catches this.
    """
    raw = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "pinning_tests",
        "evidence": "the code is fragile",
        "prescription": "fix it",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw)
    assert any("does not cite" in p for p in problems)


def test_run_file_citation_accepted_when_exists_refused_when_not(tmp_path):
    """T4: A run-file citation is accepted when the file exists, and refused
    when it does not.

    Red without C1: brief.json has no colon and is rejected by the old check.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # brief.json exists -> accepted
    (run_dir / "brief.json").write_text("{}")
    raw_brief = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "brief.json",
        "prescription": "see the brief",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw_brief, "", run_dir=str(run_dir))
    assert not any("brief.json" in p for p in problems)

    # research file exists -> accepted
    (run_dir / "research").mkdir()
    (run_dir / "research" / "api-surface.md").write_text("# Findings\n")
    raw_research = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "research/api-surface.md",
        "prescription": "see the research",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw_research, "", run_dir=str(run_dir))
    assert not any("api-surface" in p for p in problems)

    # brief.json does not exist -> refused
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    problems = pitfalls_mod.validate_pitfalls(raw_brief, "", run_dir=str(empty_run))
    assert any("brief.json" in p and "not found" in p for p in problems)

    # research file does not exist -> refused
    problems = pitfalls_mod.validate_pitfalls(raw_research, "", run_dir=str(empty_run))
    assert any("api-surface" in p and "not found" in p for p in problems)


def test_absent_claim_rejects_absolute_and_escape_paths():
    """T5: Absence claims with absolute or .. paths are refused as unsafe."""
    raw_abs = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "absent: /etc/passwd",
        "prescription": "should not work",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw_abs)
    assert any("absolute" in p for p in problems)

    raw_escape = json.dumps([{
        "id": "PIT-001", "risk": "high", "category": "scope_boundaries",
        "evidence": "absent: ../../etc/passwd",
        "prescription": "should not work",
    }])
    problems = pitfalls_mod.validate_pitfalls(raw_escape)
    assert any("escapes" in p for p in problems)


def test_a_pitfall_answered_under_its_own_heading_is_quoted_with_the_block_under_it():
    """dm-219: a ``### PIT-001`` entry's answer is the block under its heading."""
    design = _contract(anti="### PIT-001 (high) - Verifier format\n\n"
                            "The verifier reads it at `file.py:1`.\n\n"
                            "### PIT-002 (low)\n\nOther.")
    assert _citations(design, _pit(), _pit("PIT-002", "low")) == {
        "PIT-001": [("Anti-patterns to avoid",
                     "PIT-001 (high) - Verifier format The verifier reads it at `file.py:1`.")],
        "PIT-002": [("Anti-patterns to avoid", "PIT-002 (low) Other.")]}


def test_a_cross_reference_under_a_pitfalls_heading_stays_in_its_quote():
    """Review of #519: "Unlike PIT-002, ..." is PIT-001's answer, not the start of PIT-002's."""
    design = _contract(anti="### PIT-001 (high)\n\nUnlike PIT-002, fixed at `file.py:1`.\n\n"
                            "- **PIT-002** (low): see \u00a7 Changes")
    quotes = _citations(design, _pit(), _pit("PIT-002", "low"))
    assert quotes["PIT-001"] == [("Anti-patterns to avoid",
                                  "PIT-001 (high) Unlike PIT-002, fixed at `file.py:1`.")]
