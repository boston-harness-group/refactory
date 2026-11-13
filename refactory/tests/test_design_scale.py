"""The design phase scales with the change, and its review blocks only on what changes the build.

A scoped change (a refine, or one whose research footprint shows \u22642 existing functions
touched \u2014 counted by (path, name) \u2014 nothing new added, no shared engine changed) gets at
most five pitfalls, one strategy perspective.  Non-refine research is guided to 3 to 5 directions;
classify_scale runs after research and sets the class from the footprint.  At every
scale the design goes straight from the claims check to the design-review gate, which
sends it back only when its ## Changes section is missing or empty, and once; everything
else -- a misquote, a citation past the end of its file, a high-risk pitfall the
design never mentions -- is listed as advisory at the person's gate and costs no round,
and an optional section a design leaves out is not checked.  Pitfalls are quoted on the
card, never judged (dm-262).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from refactory_dsh.design import DesignMode
from refactory_dsh.design_ops import (
    REFINE_SCOPE_REASON,
    classify_scale,
    design_review_status,
    gate_design_review,
)
from test_design_contract_extensions import FakeCtx, _make_design

def _gate(design: str, *, brief: str = "a tool", scope: str = "full", source: str = "",
          pitfalls: list | None = None):
    files = {"design.md": design, "brief.json": json.dumps({"brief": brief})}
    if pitfalls is not None:
        files["pitfalls.json"] = json.dumps(pitfalls)
    ctx = FakeCtx(
        files=files,
        facts={"name": "scale_test", "has_pitfalls_step": pitfalls is not None,
               "research_scope": scope, "source": source},
    )
    return ctx, gate_design_review(ctx)


# ── what blocks ────────────────────────────────────────────────────────────────

def test_only_a_missing_or_empty_changes_section_sends_the_design_back():
    ctx, verdict = _gate(_make_design().replace("## Changes", "## Notes"))
    assert verdict == "reloop"
    assert "the build needs, or it is empty: ## Changes" in verdict.reason
    # A heading with nothing under it is as missing as no heading.
    ctx, verdict = _gate(_make_design().replace("- tool.py\n", ""))
    assert verdict == "reloop" and "## Changes" in verdict.reason
    # Rule and sites is optional now: a design without it goes to the person.
    ctx, verdict = _gate(_make_design().replace("## Rule and sites", "## Notes"))
    assert verdict == "proceed"


# ── what is advisory ──────────────────────────────────────────────────────────

def test_a_light_design_passes_with_nothing_to_list():
    light = ("# Make the tool print what it built\n\n## Summary\n\nIt prints nothing now.\n\n"
             "## Changes\n\n- `tool.py` \u2014 print the built name\n\n"
             "## Tests\n\n- `tests/test_tool.py::test_build` \u2014 the name is printed\n")
    ctx, verdict = _gate(light, pitfalls=[
        {"id": "PIT-002", "risk": "low", "text": "the log line moves"}])
    assert verdict == "proceed"
    assert ctx.facts["design_review_advisory"] == ""
    assert ctx.facts["design_review_status"] == "Design review: PASS"
    # A low-risk pitfall the design does not mention is listed, not advised on.
    assert ctx.facts["pitfall_coverage"] == "Pitfall citations:\n  PIT-002 (low): not mentioned"


def test_a_light_design_names_the_high_risk_pitfalls_it_never_mentions():
    """The card says which high-risk pitfalls the design never cites, in one line."""
    light = ("# Make the tool print what it built\n\n## Summary\n\nIt prints nothing now.\n\n"
             "## Changes\n\n- `tool.py` \u2014 print the built name\n\n"
             "## Tests\n\n- `tests/test_tool.py::test_build` \u2014 the name is printed\n")
    ctx, verdict = _gate(light, pitfalls=[
        {"id": "PIT-001", "risk": "high", "text": "the cache key ignores the base"},
        {"id": "PIT-002", "risk": "low", "text": "the log line moves"},
        {"id": "PIT-003", "risk": "high", "text": "a caller passes None"}])
    assert verdict == "proceed"
    assert ctx.facts["design_review_status"] == (
        "Design review: PASS\nOther advisories: 1\n"
        "- high-risk pitfall(s) the design does not mention: PIT-001, PIT-003")


def test_a_missing_summary_or_tests_section_is_listed():
    ctx, verdict = _gate(_make_design().replace("## Tests", "## Notes")
                         .replace("## Summary", "## Notes"))
    assert verdict == "proceed"
    assert "the design is missing section(s): Summary, Tests" in ctx.facts["design_review_advisory"]


def test_a_missing_ask_mapping_or_failure_trace_is_not_checked():
    ctx, verdict = _gate(_make_design(ask_mapping=False, failure_trace=False))
    assert verdict == "proceed"
    assert ctx.facts["design_review_advisory"] == ""


def test_a_citation_past_the_end_of_its_file_is_listed(tmp_path):
    (tmp_path / "tool.py").write_text("x = 1\ny = 2\n")
    design = _make_design().replace("## Changes", "`x` at `tool.py:40`: `x = 1`\n\n## Changes")
    ctx, verdict = _gate(design, source=str(tmp_path))
    assert verdict == "proceed"
    assert "tool.py has 2 lines, not 40" in ctx.facts["design_review_advisory"]


def test_a_misquote_of_the_brief_is_listed():
    design = _make_design() + '\nThe brief says "remove the flag from the command".\n'
    ctx, verdict = _gate(design, brief="Add the flag to the command.")
    assert verdict == "proceed"
    assert "does not appear verbatim in the brief" in ctx.facts["design_review_advisory"]


def test_a_misquote_that_drops_a_negation_is_not_called_a_punctuation_slip():
    # "able" is inside "unable": a word-level match must not accept the inverted quote.
    design = _make_design() + '\nThe brief says "Able to overwrite files".\n'
    ctx, verdict = _gate(design, brief="The tool is unable to overwrite files.")
    assert verdict == "proceed"
    assert "does not appear verbatim in the brief" in ctx.facts["design_review_advisory"]
    assert "only in punctuation" not in ctx.facts["design_review_advisory"]


def test_a_high_risk_pitfall_the_design_never_mentions_is_listed():
    design = _make_design() + "\n## Anti-patterns to avoid\n\n- PIT-002: addressed at \u00a7 Changes\n"
    ctx, verdict = _gate(design, pitfalls=[
        {"id": "PIT-001", "risk": "high", "text": "the cache key ignores the base"}])
    assert verdict == "proceed"
    assert "high-risk pitfall(s) the design does not mention: PIT-001" in (
        ctx.facts["design_review_advisory"])


def test_a_quote_differing_only_in_punctuation_and_case_is_advisory():
    design = _make_design() + '\nThe brief says "add the **flag** to the command".\n'
    ctx, verdict = _gate(design, brief="Add the flag to the command.")
    assert verdict == "proceed"
    assert "differs from the brief only in punctuation" in ctx.facts["design_review_advisory"]
    assert ctx.facts["design_review_status"].splitlines()[0] == "Design review: PASS"
    assert "Other advisories: 1" in ctx.facts["design_review_status"]


def test_paperwork_items_are_listed_and_cost_no_round():
    """b35a's gate read 'FAIL, 33 unresolved' on items like these; now it reads PASS."""
    design = (_make_design(failure_trace=False)
              .replace("# Fix the thing", "# scale_test")
              .replace("## Rule and sites\n\n- `tool.py:1` \u2014 **[changed]**",
                       "## Rule and sites\n\n- `tool.py:1` \u2014 **[changed]**\n- `other.py:3` \u2014 the caller"))
    ctx, verdict = _gate(design)
    assert verdict == "proceed"
    advisory = ctx.facts["design_review_advisory"]
    assert "Failure trace" not in advisory  # optional: not checked
    assert "title is the run name" in advisory
    assert "other.py:3" in advisory
    status = ctx.facts["design_review_status"]
    assert status.splitlines()[0] == "Design review: PASS"
    assert "Other advisories: 2" in status and "other.py:3" in status


def test_an_unmentioned_medium_pitfall_is_listed_but_not_advised_on():
    design = _make_design() + "\n## Anti-patterns to avoid\n\n- none of high risk\n"
    ctx, verdict = _gate(design, pitfalls=[
        {"id": "PIT-002", "risk": "medium", "text": "a slower path on large inputs"}])
    assert verdict == "proceed"
    assert "PIT-002 (medium): not mentioned" in ctx.facts["pitfall_coverage"]
    assert "pitfall" not in ctx.facts["design_review_advisory"]


def test_the_fail_count_counts_only_blocking_findings():
    # The findings text a person sees can carry other bulleted lines; the count comes
    # from the blocking list alone.
    facts = {"design_review_attempts_used": 1,
             "design_review_findings": "- a blocking finding\n- a listed note\n- another note",
             "design_review_blocking": "- a blocking finding",
             "design_review_advisory": "- a note\n- another note"}
    assert design_review_status(facts) == (
        "Design review: FAIL, 1 unresolved\n- a blocking finding\n"
        "Other advisories: 2\n- a note\n- another note")


# ── rounds ────────────────────────────────────────────────────────────────────

def test_every_scale_gets_one_rewrite_for_a_missing_section():
    design = _make_design().replace("## Changes", "## Notes")
    for scope in ("full", "scoped"):
        ctx, first = _gate(design, scope=scope)
        assert first == "reloop", scope
        second = gate_design_review(ctx)
        assert second == "proceed", scope
        assert ctx.facts["design_review_status"].startswith("Design review: FAIL, 1 unresolved")
        assert "## Changes" in ctx.facts["design_review_findings"]


# ── the prompts scale with the change ─────────────────────────────────────────

def test_the_design_phase_prompts_scale_with_the_change():
    nodes = DesignMode().build().nodes
    assert "one focused question for a scoped change" in nodes["research_director"].prompt_template
    assert "{{research_scope}}" in nodes["pitfalls_researcher"].prompt_template
    assert "list at most five" in nodes["pitfalls_researcher"].prompt_template
    # A scoped change has no perspectives step: route_strategy sends it past
    # strategy_director (test_design_short_path.py).
    graph = DesignMode().build()
    assert graph.out_edges("route_strategy", condition="scoped")[0].target == "synthesize_strategy"


def test_the_design_does_not_account_for_every_finding():
    nodes = DesignMode().build().nodes
    assert "Carry every finding id forward" not in nodes["synthesize_strategy"].prompt_template
    design_doc = nodes["design_doc"].prompt_template
    assert "neither addressed nor set aside is blocking" not in design_doc
    assert "where it changes a decision" in design_doc
    # Nothing reviews the design before the person does.
    assert "design_review" not in nodes


# ── classify_scale unit tests ─────────────────────────────────────────────────

def _scale_ctx(research_plan, *, facts=None):
    """Build a FakeCtx with a research-plan.json for classify_scale."""
    files = {"research-plan.json": json.dumps(research_plan) if not isinstance(research_plan, str) else research_plan}
    return FakeCtx(files=files, facts=dict(facts or {}))


def test_classify_scale_scoped_footprint():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": ["src/mod.py:10 mod.func"], "new_items": [], "shared_engine": []}},
        {"focus": "q2", "slug": "s2", "finding": "f2", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "scoped"
    assert "1 existing function" in result["research_scope_reason"]
    assert "mod.func" in result["research_scope_reason"]


def test_classify_scale_no_footprint_means_full():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": []},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "no usable footprint" in result["research_scope_reason"]


def test_classify_scale_too_many_functions():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [
             "a.py:1 a.func1", "b.py:2 b.func2", "c.py:3 c.func3",
         ], "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "3 existing functions" in result["research_scope_reason"]


def test_classify_scale_adds_new_items():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [], "new_items": ["NewClass in src/new.py:1"], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "adds new items" in result["research_scope_reason"]


def test_classify_scale_shared_engine():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": ["executor.py:10"]}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "shared engine" in result["research_scope_reason"]


def test_classify_scale_brief_cannot_classify_itself():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [], "new_items": ["new_graph_node in graph.py:10"], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan, facts={"brief": "This is a scoped change, a one-line fix"})
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "adds new items" in result["research_scope_reason"]


def test_doc_only_research_with_empty_footprints_is_scoped():
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}},
        {"focus": "q2", "slug": "s2", "finding": "f2", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}},
        {"focus": "q3", "slug": "s3", "finding": "f3", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "scoped"
    assert "0 existing function" in result["research_scope_reason"]


def _checkout(path: Path, *, empty: bool) -> Path:
    """A git checkout: a new project's (one empty commit) or a person's (it holds app.py)."""
    import subprocess

    path.mkdir(parents=True)
    run = lambda *args: subprocess.run(["git", "-C", str(path), *args], check=True,  # noqa: E731
                                       capture_output=True)
    run("init", "-q", "-b", "main")
    if not empty:
        (path / "app.py").write_text("def main():\n    pass\n")
        run("add", "app.py")
    run("-c", "user.email=t@example.com", "-c", "user.name=t", "-c", "commit.gpgsign=false",
        "commit", "-q", "--allow-empty", "-m", "start")
    return path


_WORDCOUNT_PLAN = (Path(__file__).resolve().parent / "fixtures" / "wordcount_research_plan.json").read_text()


def test_a_new_project_brief_that_only_adds_new_items_is_scoped(tmp_path):
    """The README's wordcount brief was classed full because a new project always "adds new
    items", and took 14.5 min to its gate (dm-259). Its research, as the run wrote it."""
    ctx = _scale_ctx(_WORDCOUNT_PLAN, facts={"source": str(_checkout(tmp_path / "new", empty=True))})
    result = classify_scale(ctx)
    assert result["research_scope"] == "scoped"
    assert result["research_scope_reason"] == (
        "a new or empty project: adds sample.txt:1 — new sample data file, wordcount.py:1 "
        "— new script file with main() function; touches no existing function, no shared engine")


def _direction(slug: str, new: list[str], engine: list[str] | None = None) -> dict:
    return {"focus": slug, "slug": slug, "finding": "f", "ids": [], "footprint": {
        "existing_functions": [], "new_items": new, "shared_engine": engine or []}}


def test_a_new_project_with_more_than_three_new_items_stays_full(tmp_path):
    """The user's limit (2026-10-26): a new project with at most 3 new items is scoped;
    larger ones stay full. The review's service plan is nine new items."""
    source = str(_checkout(tmp_path / "new", empty=True))
    service = [
        _direction("api", ["api/server.py:1 FastAPI app with 12 REST endpoints",
                           "api/auth.py:1 OAuth2 + JWT middleware"]),
        _direction("storage", ["db/models.py:1 SQLAlchemy models for 9 tables",
                               "db/migrations/:1 Alembic migrations"]),
        _direction("worker", ["worker/queue.py:1 Redis-backed job queue",
                              "worker/tasks.py:1 6 background tasks"]),
        _direction("cli", ["cli/main.py:1 admin CLI with 8 commands"]),
        _direction("deploy", ["Dockerfile:1 image", "docker-compose.yml:1 api+worker+db+redis"]),
    ]
    result = classify_scale(_scale_ctx(service, facts={"source": source}))
    assert result["research_scope"] == "full"
    assert result["research_scope_reason"].startswith(
        "a new or empty project that adds 9 new items (max 3 for scoped): ")

    items = ["a.py:1 a", "b.py:1 b", "c.py:1 c", "d.py:1 d"]
    three = classify_scale(_scale_ctx([_direction("q", items[:3])], facts={"source": source}))
    assert three == {"research_scope": "scoped", "research_scope_reason":
                     "a new or empty project: adds a.py:1 a, b.py:1 b, c.py:1 c; touches no "
                     "existing function, no shared engine"}
    # Counted once however many directions name an item.
    twice = [_direction("q", items[:3]), _direction("r", items[1:3])]
    assert classify_scale(_scale_ctx(twice, facts={"source": source}))["research_scope"] == "scoped"
    four = classify_scale(_scale_ctx([_direction("q", items)], facts={"source": source}))
    assert four["research_scope"] == "full"


def test_an_existing_project_brief_that_adds_new_items_is_still_full(tmp_path):
    ctx = _scale_ctx(_WORDCOUNT_PLAN, facts={"source": str(_checkout(tmp_path / "old", empty=False))})
    assert classify_scale(ctx) == {
        "research_scope": "full",
        "research_scope_reason": "adds new items: sample.txt:1 — new sample data file, "
                                 "wordcount.py:1 — new script file with main() function"}


def test_a_new_project_still_goes_full_on_a_shared_engine_or_a_cited_function(tmp_path):
    source = str(_checkout(tmp_path / "new", empty=True))
    engine = [{"focus": "q", "slug": "s", "finding": "f", "ids": [], "footprint": {
        "existing_functions": [], "new_items": ["tool.py:1 new"], "shared_engine": ["executor.py:10"]}}]
    assert classify_scale(_scale_ctx(engine, facts={"source": source}))["research_scope_reason"] == (
        "changes a shared engine: executor.py:10")
    # Nothing exists to cite in an empty checkout: a cited function is unresolvable.
    cited = [{"focus": "q", "slug": "s", "finding": "f", "ids": [], "footprint": {
        "existing_functions": ["app.py:1 main"], "new_items": ["tool.py:1 new"], "shared_engine": []}}]
    assert classify_scale(_scale_ctx(cited, facts={"source": source}))["research_scope"] == "full"
    # Without a checkout to read, nothing says the project is empty: full, as before.
    assert classify_scale(_scale_ctx(_WORDCOUNT_PLAN))["research_scope"] == "full"


def test_footprint_entry_without_evidence_is_full():
    # A bare string without path:line evidence
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": ["module.function"], "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "footprint without evidence" in result["research_scope_reason"]

    # A new_items entry without path:line
    plan2 = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [], "new_items": ["SomeClass"], "shared_engine": []}},
    ]
    ctx2 = _scale_ctx(plan2)
    result2 = classify_scale(ctx2)
    assert result2["research_scope"] == "full"
    assert "adds new items" in result2["research_scope_reason"]


def test_footprint_nonexistent_path_is_full(tmp_path):
    """Correction #9: a footprint citing a path that does not exist is classed full."""
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": ["totally/nonexistent/file.py:10 some_func"],
                       "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan, facts={"source": str(tmp_path)})
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "footprint without evidence" in result["research_scope_reason"]
    assert "nonexistent" in result["research_scope_reason"]

    # A path that exists is accepted
    (tmp_path / "real_file.py").write_text("def func(): pass\n")
    plan2 = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": ["real_file.py:1 func"],
                       "new_items": [], "shared_engine": []}},
    ]
    ctx2 = _scale_ctx(plan2, facts={"source": str(tmp_path)})
    result2 = classify_scale(ctx2)
    assert result2["research_scope"] == "scoped"


def test_null_footprint_field_is_full():
    """A footprint with null existing_functions or new_items degrades to full, not crash."""
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": None, "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "no usable footprint" in result["research_scope_reason"]

    plan2 = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [], "new_items": None, "shared_engine": []}},
    ]
    ctx2 = _scale_ctx(plan2)
    result2 = classify_scale(ctx2)
    assert result2["research_scope"] == "full"
    assert "no usable footprint" in result2["research_scope_reason"]


def test_a_shared_engine_that_is_not_a_list_is_full():
    """shared_engine is a list of citations. A bare string, a number, an object,
    None, True or False is unreadable, so the run is full; only [] means none."""
    for value in ("refactory_dsh/nn/executor.py:10", 3, {"at": "x.py:1"}, None, True, False):
        plan = [
            {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
             "footprint": {"existing_functions": [], "new_items": [], "shared_engine": value}},
        ]
        result = classify_scale(_scale_ctx(plan))
        assert result["research_scope"] == "full", value
        assert "shared_engine is not a list" in result["research_scope_reason"], value
    plan = [{"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
             "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}}]
    assert classify_scale(_scale_ctx(plan))["research_scope"] == "scoped"


def _one(footprint, *, facts=None):
    """classify_scale on a one-direction plan with *footprint*."""
    plan = [{"focus": "q1", "slug": "s1", "finding": "f1", "ids": [], "footprint": footprint}]
    return classify_scale(_scale_ctx(plan, facts=facts))


def _fns(*entries):
    return {"existing_functions": list(entries), "new_items": [], "shared_engine": []}


def test_a_footprint_needs_exactly_its_three_keys():
    for footprint in (
        {},
        {"shared_engine": []},
        {"existingFunctions": ["a.py:1 f", "a.py:2 g", "a.py:3 h", "a.py:4 i"],
         "new_items": [], "shared_engine": []},
        {"existing_functions": [], "new_items": [], "shared_engines": ["executor.py:1"]},
        {"existing_functions": [], "new_items": [], "shared_engine": [],
         "touched_functions": ["a.py:1 f", "a.py:2 g", "a.py:3 h"]},
    ):
        result = _one(footprint)
        assert result["research_scope"] == "full", footprint
        assert "no usable footprint" in result["research_scope_reason"], footprint


def test_a_function_citation_needs_a_line_number_and_one_name():
    for entry in ("a.py: foo", "a.py:abc foo", "a.py:", "a.py:1", "a.py:0 foo",
                  "a.py:1 f, g, h, k", "module.function"):
        result = _one(_fns(entry))
        assert result["research_scope"] == "full", entry
        assert "footprint without evidence" in result["research_scope_reason"], entry
    assert _one(_fns("a.py:12 Mode.build"))["research_scope"] == "scoped"
    # A line range is a citation too, counted by the line it starts on.
    result = _one(_fns("a.py:2101-2141 blocking_findings"))
    assert result["research_scope"] == "scoped"
    assert "a.py:2101" in result["research_scope_reason"]
    assert _one(_fns("a.py:5-x f"))["research_scope"] == "full"


def test_a_citation_outside_the_checkout_never_counts(tmp_path):
    (tmp_path / "x.py").write_text("def f():\n    pass\n")
    facts = {"source": str(tmp_path)}
    for entry in ("/etc/hosts:1 foo", "../x.py:1 f", "~/x.py:1 f", ("a" * 300) + ".py:1 f"):
        result = _one(_fns(entry), facts=facts)
        assert result["research_scope"] == "full", entry
    assert _one(_fns("x.py:1 f"), facts=facts)["research_scope"] == "scoped"


def test_a_tracked_symlink_out_of_the_checkout_names_nothing(tmp_path):
    """The shared resolver follows symlinks: a tracked one whose target is outside the
    checkout names no checkout file, so the gate never reads the target either."""
    import os
    import subprocess

    from refactory_dsh.design_ops import cited_candidates, resolve_cited, tracked_files

    (tmp_path / "outside.py").write_text("a = 1\nb = 2\n")
    source = tmp_path / "src"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "inside.py").write_text("c = 3\n")
    os.symlink(tmp_path / "outside.py", source / "pkg" / "link.py")
    os.symlink(source / "pkg" / "inside.py", source / "alias.py")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    tracked = tracked_files(str(source))
    assert "pkg/link.py" in tracked
    for cited in ("pkg/link.py", "link.py"):
        assert cited_candidates(str(source), cited, tracked) == [], cited
        assert resolve_cited(str(source), cited, tracked) is None, cited
    # A symlink that stays inside the checkout is a checkout file.
    assert resolve_cited(str(source), "alias.py", tracked) == "alias.py"
    # The gate's citation check does not report the outside file's length.
    design = _make_design().replace("## Changes", "`pkg/link.py:40` is cited.\n\n## Changes")
    ctx, verdict = _gate(design, source=str(source))
    assert verdict == "proceed"
    assert "lines, not 40" not in ctx.facts["design_review_advisory"]


def test_citations_resolve_against_the_real_checkout_however_research_writes_them():
    """Research cites from the repository root, a package, or a bare file name.
    Each resolves against the checkout's tracked files; a file that is not there,
    or a name two files share, does not."""
    repo = Path(__file__).resolve().parents[2]
    facts = {"source": str(repo)}
    for entry in ("refactory/refactory_dsh/create_ops.py:324 _plan",
                  "refactory_dsh/create_ops.py:324 _plan",
                  "create_ops.py:324 _plan"):
        result = _one(_fns(entry), facts=facts)
        assert result["research_scope"] == "scoped", (entry, result)
        assert "refactory/refactory_dsh/create_ops.py:324" in result["research_scope_reason"]
    assert _one(_fns("no_such_module_zz.py:1 f"), facts=facts)["research_scope"] == "full"
    # A suffix counts only on a whole path segment: "eate_ops.py" is not create_ops.py.
    assert _one(_fns("eate_ops.py:324 _plan"), facts=facts)["research_scope"] == "full"
    assert _one(_fns("__init__.py:1 f"), facts=facts)["research_scope"] == "full"


def test_same_named_functions_in_one_file_count_separately():
    result = _one(_fns("fast.py:10 build", "fast.py:200 build", "fast.py:300 build"))
    assert result["research_scope"] == "full"
    assert "touches 3 existing functions" in result["research_scope_reason"]


def test_two_functions_stay_scoped_and_three_do_not():
    assert _one(_fns("a.py:1 f", "b.py:2 g"))["research_scope"] == "scoped"
    assert _one(_fns("a.py:1 f", "b.py:2 g", "c.py:3 h"))["research_scope"] == "full"


def test_a_rerun_reclassifies_a_scoped_run_from_its_footprint():
    """A non-refine run that an earlier visit classed scoped is classed again from
    its footprint: research_scope, which classify_scale writes, is never read."""
    facts = {"research_scope": "scoped",
             "research_scope_reason": "touches 1 existing function(s), adds nothing new, no shared engine"}
    result = _one(_fns("a.py:1 f", "b.py:2 g", "c.py:3 h"), facts=facts)
    assert result["research_scope"] == "full"


def test_a_plan_in_prose_and_fences_is_read():
    raw = ("Here is the plan:\n```json\n"
           + json.dumps([{"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
                          "footprint": _fns("a.py:1 f")}])
           + "\n```\nDone.")
    result = classify_scale(_scale_ctx(raw))
    assert result["research_scope"] == "scoped"
    assert "a.py:1" in result["research_scope_reason"]


def test_a_direction_that_is_not_an_object_is_full():
    plan = [{"focus": "q1", "slug": "s1", "finding": "f1", "ids": [], "footprint": _fns()},
            "a stray string"]
    result = classify_scale(_scale_ctx(plan))
    assert result["research_scope"] == "full"
    assert "not an object" in result["research_scope_reason"]


def test_a_failing_full_design_review_shows_the_scale():
    status = design_review_status({
        "design_review_attempts_used": 1,
        "design_review_findings": "- x.py: something",
        "design_review_blocking": "- x.py: something",
        "research_scope_reason": "touches 3 existing functions (max 2 for scoped): f, g, h",
    })
    assert "FAIL" in status
    assert "Scale: full" in status
    assert "touches 3 existing functions" in status

def test_empty_directions_is_full():
    """An empty research plan (no directions) is classed full."""
    ctx = _scale_ctx([])
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "no usable footprint" in result["research_scope_reason"]


def test_classify_scale_reason_in_design_review_status():
    # scoped
    scoped_facts = {
        "design_review_attempts_used": 1,
        "design_review_findings": "(none: the design review passed)",
        "research_scope": "scoped",
        "research_scope_reason": "touches 1 existing function(s) (f at x.py:1), adds nothing new, no shared engine",
    }
    status = design_review_status(scoped_facts)
    assert "Scale: scoped" in status
    assert "touches 1 existing function" in status
    assert "PASS" in status

    # full
    full_facts = {
        "design_review_attempts_used": 1,
        "design_review_findings": "(none: the design review passed)",
        "research_scope_reason": "adds new items: NewClass in src/new.py:1",
    }
    status = design_review_status(full_facts)
    assert "Scale: full" in status
    assert "adds new items" in status

    # refine
    refine_facts = {
        "design_review_attempts_used": 1,
        "design_review_findings": "(none: the design review passed)",
        "research_scope": "scoped",
        "research_scope_reason": REFINE_SCOPE_REASON,
    }
    status = design_review_status(refine_facts)
    assert "Scale: scoped" in status
    assert "refinement" in status


# ── _plan output pinning ──────────────────────────────────────────────────────

def test_plan_output_on_existing_research_fixture():
    """_plan strips footprint and returns only focus/slug/finding/ids."""
    from refactory_dsh.create_ops import _plan
    from tests_support import RESEARCH_ANSWER

    entries = _plan(RESEARCH_ANSWER)
    assert len(entries) == 3
    for entry in entries:
        assert set(entry.keys()) == {"focus", "slug", "finding", "ids"}
        assert "footprint" not in entry


# ── end-to-end tests ──────────────────────────────────────────────────────────

def _design_run(tmp_path, brief, answers, *, task_extra=None):
    from tests_support import FakeModel, git_repo

    repo = git_repo(tmp_path / "repo")
    model = FakeModel(answers)
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    task = {"name": "scale_tool", "brief": brief,
            "check": f"{sys.executable} tool.py", "target": str(repo)}
    if task_extra:
        task.update(task_extra)
    run = mode.run(tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
                   task=task)
    return run, model


_FULL_BUILD = [
    "perspectives",
    "one plan",
]


def test_structural_brief_naming_files_is_full(tmp_path):
    from tests_support import (
        DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
        RESEARCH_ANSWER, VERIFICATION_ANSWER, render,
    )

    brief = (
        "Add a graph node dispute_resolver to design.py and design_ops.py and "
        "wire it into all six graphs across both runtimes"
    )
    build = [render({"tool.py": "def build():\n    print('built')\n"}),
             "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]
    run, model = _design_run(tmp_path, brief, [
        RESEARCH_ANSWER, PITFALLS_ANSWER, *_FULL_BUILD, DESIGN_DOC_ANSWER,
        *build])

    assert run.exit_ok, run.error
    assert run.facts["research_scope"] == "full"
    assert "no usable footprint" in run.facts["research_scope_reason"]
    # perspectives ran
    assert any("strategy director" in p.lower() for p in model.prompts)


def test_one_line_doc_fix_stays_scoped(tmp_path):
    from tests_support import (
        DESIGN_DOC_ANSWER, PITFALLS_ANSWER, SCOPED_RESEARCH_ANSWER,
        VERIFICATION_ANSWER, render,
    )

    brief = "Add one sentence to the live-control doc and update refactory/docs/live-control.md"
    build = [render({"tool.py": "def build():\n    print('built')\n"}),
             "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]
    run, model = _design_run(tmp_path, brief, [
        SCOPED_RESEARCH_ANSWER, PITFALLS_ANSWER, "one plan", DESIGN_DOC_ANSWER, *build])

    assert run.exit_ok, run.error
    assert run.facts["research_scope"] == "scoped"
    assert "adds nothing new" in run.facts["research_scope_reason"]
    assert not any("strategy director" in p.lower() for p in model.prompts)


def test_brief_calling_itself_scoped_is_still_full(tmp_path):
    from tests_support import (
        DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
        RESEARCH_ANSWER, VERIFICATION_ANSWER, render,
    )

    brief = "This is a scoped change, a one-line fix: add a new graph node to design.py"
    build = [render({"tool.py": "def build():\n    print('built')\n"}),
             "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]
    run, model = _design_run(tmp_path, brief, [
        RESEARCH_ANSWER, PITFALLS_ANSWER, *_FULL_BUILD, DESIGN_DOC_ANSWER,
        *build])

    assert run.exit_ok, run.error
    assert run.facts["research_scope"] == "full"


def test_refine_run_stays_scoped(tmp_path, monkeypatch):
    """A refine run is scoped from the start: prepare decides it (and route_refine_design
    then skips research and classify_scale)."""
    from refactory_dsh import delivery, design_ops, nn, repo_change, sandbox

    monkeypatch.setattr(repo_change, "wants_worktree", lambda ctx: False)
    monkeypatch.setattr(repo_change, "locate",
                        lambda ctx, name, task, **kw: {
                            "source": str(tmp_path), "check_root": str(tmp_path),
                            "artifact": str(tmp_path), "branch": "", "change": "",
                        })
    monkeypatch.setattr(sandbox, "model_readiness", lambda *a, **kw: {})
    monkeypatch.setattr(delivery, "plan_run",
                        lambda ctx, *, task: {"delivery": "test", "delivery_plan": {}})
    workdir = tmp_path / "run"
    task = {"name": "polish_ui", "brief": "Fix the spacing in the widget",
            "check": "python -m pytest -q",
            "refine": "https://github.com/example/repo/pull/1"}
    ctx = nn.RunContext(workdir, task=task)
    result = design_ops.prepare_design(ctx, deliver=True, in_worktree=False)
    assert result["research_scope"] == "scoped"
    assert result["research_scope_reason"] == REFINE_SCOPE_REASON


def test_no_footprint_means_full_end_to_end(tmp_path):
    from tests_support import (
        DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
        RESEARCH_ANSWER, VERIFICATION_ANSWER, render,
    )

    build = [render({"tool.py": "def build():\n    print('built')\n"}),
             "health: ok", "code review: ok", "qa: ok", VERIFICATION_ANSWER, "VERDICT: PASS"]
    run, model = _design_run(tmp_path, "Improve the logging.", [
        RESEARCH_ANSWER, PITFALLS_ANSWER, *_FULL_BUILD, DESIGN_DOC_ANSWER,
        *build])

    assert run.exit_ok, run.error
    assert run.facts["research_scope"] == "full"
    assert "no usable footprint" in run.facts["research_scope_reason"]


# ── spine op entry test ───────────────────────────────────────────────────────

def test_classify_scale_through_run(tmp_path):
    """Run classify_scale through refactory_dsh.run's build_op_context."""
    from refactory_dsh.run import build_op_context

    workdir = tmp_path / "run"
    workdir.mkdir()
    # Write declarations the runner reads
    decl_dir = workdir / ".refactory"
    decl_dir.mkdir()
    (decl_dir / "declarations.json").write_text(json.dumps({
        "classify_scale": {"reads": ["research-plan.json"], "writes": [], "produces": ["research_scope", "research_scope_reason"]}
    }))
    (decl_dir / "knobs.json").write_text("{}")

    plan = [{"focus": "q", "slug": "s", "finding": "f", "ids": []}]

    # Case 1: scoped footprint
    facts_full = {"research_scope": "full", "research_scope_reason": ""}
    (workdir / "facts.json").write_text(json.dumps(facts_full))
    plan_scoped = [{"focus": "q", "slug": "s", "finding": "f", "ids": [],
                    "footprint": {"existing_functions": ["x.py:1 fn"], "new_items": [], "shared_engine": []}}]
    (workdir / "research-plan.json").write_text(json.dumps(plan_scoped))
    ctx2 = build_op_context(workdir, "classify_scale")
    result2 = classify_scale(ctx2)
    assert result2["research_scope"] == "scoped"

    # Case 2: missing footprint
    (workdir / "facts.json").write_text(json.dumps(facts_full))
    (workdir / "research-plan.json").write_text(json.dumps(plan))
    ctx3 = build_op_context(workdir, "classify_scale")
    result3 = classify_scale(ctx3)
    assert result3["research_scope"] == "full"


# ── dm-149-like fixture is scoped ────────────────────────────────────────────

def test_dm149_like_fixture_is_scoped():
    """One parser function changed in a module several gates call, plus new tests."""
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": ["refactory_dsh/create_ops.py:324 _plan"],
                       "new_items": [], "shared_engine": []}},
        {"focus": "q2", "slug": "s2", "finding": "f2", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}},
        {"focus": "q3", "slug": "s3", "finding": "f3", "ids": [],
         "footprint": {"existing_functions": [], "new_items": [], "shared_engine": []}},
    ]
    # Against the real checkout, as a run classifies it, not a vacuous no-source check.
    ctx = _scale_ctx(plan, facts={"source": str(Path(__file__).resolve().parents[2])})
    result = classify_scale(ctx)
    assert result["research_scope"] == "scoped"
    assert "_plan" in result["research_scope_reason"]


def test_same_named_functions_in_different_files_count_separately():
    """Three `parse` functions in three files are counted as three and classed full."""
    plan = [
        {"focus": "q1", "slug": "s1", "finding": "f1", "ids": [],
         "footprint": {"existing_functions": [
             "a.py:10 parse", "b.py:20 parse", "c.py:30 parse",
         ], "new_items": [], "shared_engine": []}},
    ]
    ctx = _scale_ctx(plan)
    result = classify_scale(ctx)
    assert result["research_scope"] == "full"
    assert "3 existing functions" in result["research_scope_reason"]


def test_research_director_footprint_instruction_clauses():
    """Each clause of the footprint instruction is pinned; reverting any one fails."""
    graph = DesignMode().build()
    prompt = graph.nodes["research_director"].prompt_template
    # Required footprint on every direction
    assert "Record a `footprint` object on every direction" in prompt
    # Evidence format for existing_functions
    assert '"path:line name"' in prompt
    # Shared-engine definition
    assert "code that every run of the target passes through" in prompt
    # Tests and docs do not count
    assert "Test files, test functions and docs are neither touched functions nor added items" in prompt
    # Size wording is a claim, not a footprint
    assert "claim to check against the code, never a footprint" in prompt
    # Exactly the three keys, one function per entry with its real line
    assert "with exactly these three keys, each a list" in prompt
    assert "one function per entry, the line its definition starts on" in prompt
    # A shared function is not an engine
    assert "A function is not an engine because several gates or modes call it" in prompt
    # What counts as a new item
    assert ("a module, class, graph node, edge, node field, record or data shape, knob, "
            "entry point, wiring helper, or any new non-test source file") in prompt
    # The instruction works for any target: it names no file of this repository
    section = prompt[prompt.index("## Footprint"):prompt.index("## What was wrong with the last plan")]
    assert "refactory" not in section
    assert not re.search(r"\b\w+\.(py|ts)\b", section)


# ── the gate card (dm-163) ──────────────────────────────────────────────────────

_CARD_DESIGN = (
    "# Make the tool print what it built\n\n## Summary\n\nThe tool prints the name it built. "
    "Nothing else changes. A third sentence the card leaves out.\n\n"
    "## Changes\n\n- `tool.py` — print the built name\n\n"
    "## Tests\n\n- `tests/test_tool.py::test_build` — the name is printed\n\n"
    "## Anti-patterns to avoid\n\n- PIT-002: see § Changes\n")
_CARD_PITFALLS = [{"id": "PIT-001", "risk": "high", "text": "the cache key ignores the base"},
                  {"id": "PIT-002", "risk": "low", "text": "the log line moves"}]


def _card(design: str, *, attempts: int = 1) -> FakeCtx:
    ctx = FakeCtx(
        files={"design.md": design, "brief.json": json.dumps({"brief": "a tool"}),
               "pitfalls.json": json.dumps(_CARD_PITFALLS)},
        facts={"name": "card", "has_pitfalls_step": True, "research_scope": "scoped",
               "research_scope_reason": "one file", "source": "",
               "claims_check": {"unverified": ["`nope.py` does not exist"], "notes": []}})
    ctx._knobs["design_review_attempts"] = attempts
    gate_design_review(ctx)
    return ctx


def test_the_gate_card_is_plain_text_summary_review_scale_advisories_then_pitfalls():
    """The design gate's card was markdown under a '## Unresolved design review findings'
    heading, with every pitfall's coverage dumped below it (dm-163). It is plain text now:
    what the design builds, the review, the scale, the advisories with unverified claims
    first, then one short line per pitfall, each item a paragraph of its own so the
    question card keeps them apart, and nothing the old card carried is lost."""
    from types import SimpleNamespace

    from refactory_dsh.nn.executor import render

    ctx = _card(_CARD_DESIGN)
    assert ctx.facts["design_card"] == (
        "Design: The tool prints the name it built. Nothing else changes.\n\n"
        "Design review: PASS, nothing blocks the build\n\n"
        "Scale: scoped \u2014 one file\n\n"
        "Unverified claims (high, advisory): 1\n"
        "- unverified claim: `nope.py` does not exist\n\n"
        "Other advisories: 1\n"
        "- high-risk pitfall(s) the design does not mention: PIT-001\n\n"
        "Pitfalls: 2\n"
        "- PIT-001 (high): not mentioned\n"
        "- PIT-002 (low): \u00a7 Anti-patterns to avoid: \"PIT-002: see \u00a7 Changes\"")
    notes = DesignMode().build().nodes["gate_strategy"].notes
    facts = {**ctx.facts, "delivery": "this run opens a draft pull request"}
    shown = render(notes, SimpleNamespace(knobs={}, facts=facts))
    assert shown == (ctx.facts["design_card"] + "\n\nApprove the design before the artifact is written? It is "
                     "in design.md in the run directory. Then, this run opens a draft pull request; delivery.md "
                     "there says more. Answer proceed to build and deliver it that way, or reloop to have the "
                     "plan redone.")
    assert "#" not in shown

    # Out of rewrites, the unresolved section is named on the card.
    failed = _card(_CARD_DESIGN.replace("## Changes\n\n- `tool.py` \u2014 print the built name\n\n", ""),
                   attempts=0)
    assert failed.facts["design_card"].split("\n\n")[:2] == [
        "Design: The tool prints the name it built. Nothing else changes.",
        "Design review: FAIL, 1 unresolved\n"
        "- the design is missing the section the build needs, or it is empty: ## Changes"]

    # A design with no Anti-patterns section: its pitfalls are listed all the same.
    card = _card(_CARD_DESIGN.split("## Anti-patterns")[0]).facts["design_card"]
    assert card.split("\n\n")[-1] == ("Pitfalls: 2\n- PIT-001 (high): not mentioned\n"
                                       "- PIT-002 (low): not mentioned")
    assert not any(line.startswith("#") for line in card.splitlines()), "no markdown heading"
