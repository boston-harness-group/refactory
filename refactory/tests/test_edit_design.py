"""Revising a mode from a brief, and designing software from a brief.

The edit issue: the loop searches and create starts from nothing, so there was no way
to say "change this mode to do that" in words. The design issue: the third
meta-mode, which produces software rather than a package of modes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from refactory_dsh import design as design_module
from refactory_dsh import edit as edit_module
from refactory_dsh import nn, registry, sandbox
from refactory_dsh.create import CreateConfig, CreateError, CreateMode
from refactory_dsh.design import DesignMode
from refactory_dsh.repo_change import ARTIFACTS, OWNERS
from refactory_dsh.edit import EditConfig, EditMode
from refactory_dsh.run import run_node
from tests_support import (
    DESIGN_DOC_ANSWER,
    PITFALLS_ANSWER,
    RESEARCH_ANSWER,
    VERIFICATION_ANSWER,
    REVIEW_ANSWERS,
    FakeModel,
    create_answers,
    good_files,
    render,
)

NAME = "edited_pkg"
ARTIFACT = "designed_tool"


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


@pytest.fixture(autouse=True)
def _clean():
    # The owner record too: the check tests run the gate without a prepare, so a name
    # an earlier run claimed would otherwise refuse them.
    for root in (registry.workspace_root(Path(".")) / NAME, ARTIFACTS / ARTIFACT):
        shutil.rmtree(root, ignore_errors=True)
    (OWNERS / f"{ARTIFACT}.json").unlink(missing_ok=True)
    yield
    for root in (registry.workspace_root(Path(".")) / NAME, ARTIFACTS / ARTIFACT):
        shutil.rmtree(root, ignore_errors=True)
    (OWNERS / f"{ARTIFACT}.json").unlink(missing_ok=True)


def _edit_config(**overrides) -> EditConfig:
    """An edit config with the person's strategy approval recorded.

    `EditMode` is the create-v2 graph, so it carries the same `user` gate: a
    walker run stops at it, and a test that stands in for the person says so.
    """
    return EditConfig(verdicts={"gate_strategy": "proceed"}, **overrides)


def _answers() -> list[str]:
    return [*create_answers(NAME, good_files(NAME))]


# ── editing a mode ─────────────────────────────────────────────────────────
def test_the_edit_mode_is_the_create_graph_with_a_different_intent():
    """A revision is not a second graph; it is the same one with a different brief."""
    from refactory_dsh.create import CreateMode

    assert isinstance(EditMode(), CreateMode)
    assert sorted(EditMode().graph().nodes) == sorted(CreateMode().graph().nodes)
    assert EditMode.intent != CreateMode.intent
    assert "smallest edit" in EditMode.intent


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_prepare_hands_the_model_the_source_it_is_revising():
    workdir = Path(__file__).parent / "_edit_prepare"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        ctx = nn.RunContext(workdir)
        ctx.task = {"name": NAME, "brief": "keep it", "mode": "circle_packing/adaptive"}
        import refactory_dsh.create_ops as ops

        facts = ops.prepare(ctx)
        assert facts["revision_of"] == "circle_packing/adaptive"
        assert "revision of `circle_packing/adaptive`" in facts["reference"]
        assert "def greedy_pack" in facts["reference"], (
            "the source being revised must be in the prompt"
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_a_revision_needs_a_mode():
    with pytest.raises(CreateError, match="needs the mode to revise"):
        edit_module.edit(_edit_config(name=NAME, brief="anything"))


def test_a_revision_produces_a_package_and_reports_before_and_after():
    result = edit_module.edit(
        _edit_config(
            name=NAME, brief="count to a target", mode="circle_packing/adaptive",
            workdir=Path(".factory/edit-run"),
        ),
        client=FakeModel(_answers()),
    )
    assert result["revision_of"] == "circle_packing/adaptive"
    assert result["before_score"] is not None and result["after_score"] is not None
    assert result["regressed"] is False
    assert f"{NAME}/counter" in registry.render_catalogue(Path("."))


def test_the_revision_records_what_it_came_from():
    edit_module.edit(
        _edit_config(
            name=NAME, brief="count", mode="circle_packing/adaptive",
            workdir=Path(".factory/edit-run"),
        ),
        client=FakeModel(_answers()),
    )
    recorded = registry.provenance(NAME, Path("."))
    assert recorded["revision_of"] == "circle_packing/adaptive"
    assert recorded["brief"] == "count"
    assert "before_score" in recorded and "after_score" in recorded


def test_a_regression_is_refused_by_default(monkeypatch):
    monkeypatch.setattr(sandbox, "smoke", lambda identifier, *args, **kwargs: sandbox.SandboxResult(
        ok=True, payload={"exit_ok": True, "score": 5.0 if identifier == "circle_packing/adaptive" else 1.0}))
    with pytest.raises(CreateError, match="scores 1.0 where circle_packing/adaptive scores 5.0"):
        edit_module.edit(
            _edit_config(
                name=NAME, brief="count", mode="circle_packing/adaptive", attempts=1,
                workdir=Path(".factory/edit-run"),
            ),
            client=FakeModel(_answers()),
        )


def test_a_regression_can_be_intended(monkeypatch):
    monkeypatch.setattr(sandbox, "smoke", lambda identifier, *args, **kwargs: sandbox.SandboxResult(
        ok=True, payload={"exit_ok": True, "score": 5.0 if identifier == "circle_packing/adaptive" else 1.0}))
    result = edit_module.edit(
        _edit_config(
            name=NAME, brief="count", mode="circle_packing/adaptive",
            allow_regression=True, workdir=Path(".factory/edit-run"),
        ),
        client=FakeModel(_answers()),
    )
    assert result["regressed"] is None
    assert result["before_score"] is None
    assert "intentionally skipped" in result["revision_note"]


def test_a_revision_of_a_mode_that_does_not_run_is_refused(monkeypatch):
    monkeypatch.setattr(sandbox, "smoke", lambda *args, **kwargs: sandbox.SandboxResult(
        ok=True, payload={"exit_ok": False, "error": "it never ran", "score": None}))
    with pytest.raises(CreateError, match="it never ran"):
        edit_module.edit(
            _edit_config(name=NAME, brief="count", mode="circle_packing/adaptive", attempts=1),
            client=FakeModel(_answers()),
        )


# ── designing software ─────────────────────────────────────────────────────
def test_emit_writes_its_file_blocks_into_emit_txt():
    """The worker contract says not to repeat a file artifact in the final reply, so a
    prompt that asked to *answer* with file blocks got the files written to disk and a
    summary in emit.txt, which materialise cannot parse. The blocks belong in emit.txt."""
    prompt = CreateMode().build().nodes["emit"].prompt_template
    assert "Write every file into emit.txt" in prompt
    assert "Do not create the files themselves" in prompt
    assert "Answer with file blocks" not in prompt


def test_design_emit_works_in_a_clone_whose_diff_is_its_answer():
    """Design's builder no longer writes blocks into emit.txt: it works in a prepared clone
    of the last attempt, runs what it needs there, and stops; the runtime derives emit.txt
    from the clone's diff (`workspace.py`). So the block grammar (`emit_scope`), "write
    nothing but emit.txt" and the probe instruction are gone from its prompt, and its
    checklist reverts in the workspace rather than in a draft emit.txt."""
    from refactory_dsh.fast import DesignFastMode

    for mode in (DesignMode(), DesignFastMode()):
        node = mode.build().nodes["emit"]
        prompt = node.prompt_template
        assert node.workspace == {"kind": "clone", "of": "attempt", "collect": "diff"}
        assert "Your workspace is a prepared clone of the last attempt" in prompt
        assert "the workspace's diff is your answer" in prompt
        for gone in ("{{emit_scope}}", "Write those blocks into emit.txt", "Write nothing but emit.txt",
                     "{{probe_instruction}}", "Answer with file blocks"):
            assert gone not in prompt, gone
    checklist = DesignMode().build().nodes["emit"].prompt_template
    assert "revert that change in the workspace and run the same test again: it must fail" in checklist
    assert "draft emit.txt" not in checklist


def _design_answers(emit: list[str]) -> list[str]:
    """Every answer design-v2's agents need, in the order the graph asks.

    pick_corrections is a gate (no model answer needed).  When there are no
    corrections, pick_corrections returns "skip" and apply_corrections does
    not run — no model answer consumed.
    """
    return [
        # The research answer is the plan array its gate checks, for the same reason
        # create's fixture is: a prose answer is the failure the gate catches.
        RESEARCH_ANSWER,
        PITFALLS_ANSWER,
        "perspectives: the simplest thing that passes the check",
        "one plan: the files, and how the check exercises them",
        DESIGN_DOC_ANSWER,
        *emit,
        *REVIEW_ANSWERS[:3],
        VERIFICATION_ANSWER,
        REVIEW_ANSWERS[3],
    ]


def test_the_design_graph_is_a_different_shape_from_a_package_one():
    graph = DesignMode().graph()
    assert sorted(graph.nodes) == [
        "apply_corrections", "audit_corrections", "check", "check_design_claims",
        "classify_scale", "code_reviewer", "deliver",
        "delivery_gate", "design_doc", "done",
        "early_asks_check", "emit", "first_self_check", "fork_qa", "fork_self_check",
        "gate_check", "gate_design_review", "gate_overwatch",
        "gate_pitfalls", "gate_qa", "gate_research", "gate_self_check", "gate_strategy",
        "health_checker", "join_qa", "join_self_check",
        "materialise", "overwatch", "pick_corrections", "pitfalls_researcher",
        "prepare", "prepare_check", "qa_director", "qa_judge", "record_findings",
        "report", "research_director", "route_refine_design", "route_strategy",
        "strategy_director", "synthesize_qa",
        "synthesize_strategy", "verifier",
    ], "the graph is design-v2's phases, not v1's"
    assert graph.nodes["check"].call.endswith("check_beside_reviews")
    assert "register" not in graph.nodes, "an artifact is not a package"
    assert "smoke" not in graph.nodes


def test_the_design_state_validates():
    mode = DesignMode()
    assert nn.validate_state(
        mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
    ) == []


def test_designing_needs_a_check():
    """Without a check the artifact is unverified, and unverified is not a deliverable."""
    mode = DesignMode()
    result = mode.run(Path(".factory/design-nocheck"), task={"name": ARTIFACT, "brief": "x"})
    assert result.exit_ok is False
    assert "needs a check" in (result.error or "")


# prepare() checks the model credential, here a placeholder in the test's own home (dm-257).
@pytest.mark.usefixtures("configured_model_home")
def test_the_artifact_lands_beside_the_workspace_not_in_it():
    """With its worktree off, a run writes the artifact directory, as before #351."""
    ctx = nn.RunContext(Path(".factory/design-prep"), knobs={"worktree": False})
    ctx.task = {"name": ARTIFACT, "brief": "a tool", "check": "python -c 'pass'"}
    from refactory_dsh import design_ops, repo_change

    design_ops.prepare_design(ctx)
    ctx.facts["name"] = ARTIFACT
    ctx.write_text("emit.txt", render({"tool.py": "print('hi')\n"}))
    facts = repo_change.materialise(ctx)
    assert Path(facts["artifact"]).is_relative_to(ARTIFACTS.resolve())
    assert not Path(facts["artifact"]).is_relative_to(registry.workspace_root(Path(".")))


def test_the_check_gate_passes_an_artifact_that_runs():
    workdir = (ARTIFACTS / ARTIFACT).resolve()
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = nn.RunContext(Path(".factory/design-check-ctx"))
    ctx.facts.update({"name": ARTIFACT, "brief": "a checked artifact", "artifact": str(workdir), "check": "python -c \"print(1)\""})
    import refactory_dsh.design_ops as ops

    assert ops.check_artifact(ctx) == "proceed"
    assert ctx.read_json("authoring-evidence.json")["validation"]["stdout"].strip() == "1"


def test_the_check_gate_refuses_an_artifact_that_does_not_run():
    workdir = (ARTIFACTS / ARTIFACT).resolve()
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = nn.RunContext(Path(".factory/design-check-ctx2"))
    ctx.facts.update({"name": ARTIFACT, "brief": "a checked artifact", "artifact": str(workdir), "check": "python -c \"raise SystemExit(3)\""})
    import refactory_dsh.design_ops as ops

    verdict = ops.check_artifact(ctx)
    assert verdict == "reloop"
    assert "exit 3" in ctx.read_text("check.txt"), (
        "the refusal must say what the check said, or the retry is guessing"
    )
    assert "exit 3" in verdict.reason, "the CEO is told what the retry is told"


def test_the_check_runs_without_the_real_credentials(tmp_path):
    """The check runs arbitrary code a brief named, so it is sandboxed like the rest."""
    workdir = tmp_path / "artifact"
    result = sandbox.run_command(
        "python -c \"import os,sys; sys.exit(0 if 'DEEPSEEK_API_KEY' not in os.environ else 9)\"",
        workdir=workdir,
        env={**__import__("os").environ, "DEEPSEEK_API_KEY": "sk-hidden"},
    )
    assert result.ok, result.stderr


# ── design review gate tests ───────────────────────────────────────────────

_CONTRACT_DESIGN = (
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


def _contract_design(body: str = "") -> str:
    """A minimal contract-compliant design, optionally with extra body text."""
    return (
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
        + (f"\n{body}\n" if body else "")
    )


def test_design_review_gate_passes_a_clean_review(tmp_path):
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "Everything is fine.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", _CONTRACT_DESIGN)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    assert ops.gate_design_review(ctx) == "proceed"
    assert "(none: nothing blocks the build)" in ctx.facts["design_review_findings"]


def test_design_review_gate_reads_no_review(tmp_path):
    # Nothing reviews the design before the person: a review left on disk is not read.
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "BLOCKING: the design leaves alternative X open\n\nVERDICT: FAIL\n")
    ctx.write_text("design.md", _CONTRACT_DESIGN)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    assert ops.gate_design_review(ctx) == "proceed"
    assert "alternative X open" not in ctx.facts["design_review_status"]


def test_design_review_gate_catches_citation_to_missing_line(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "somefile.py").write_text("line1\nline2\nline3\n")
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(source), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", _contract_design("See somefile.py:999 for details."))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    assert ops.gate_design_review(ctx) == "proceed"
    advisory = ctx.facts["design_review_advisory"]
    assert "999" in advisory and "3 lines" in advisory


def test_design_review_gate_catches_misquoted_brief(tmp_path):
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md",
                   _contract_design('As the brief says, "enable fast mode" is required.'))
    ctx.write_json("brief.json", {"name": "x", "brief": "make it work correctly"})
    import refactory_dsh.design_ops as ops
    assert ops.gate_design_review(ctx) == "proceed"
    assert "enable fast mode" in ctx.facts["design_review_advisory"]


def test_design_review_gate_proceeds_when_attempts_exhausted(tmp_path):
    ctx = nn.RunContext(tmp_path, knobs={"design_review_attempts": 0})
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    ctx.write_text("design.md", _CONTRACT_DESIGN.replace("## Changes\n\n- file.py\n", ""))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed"
    assert "## Changes" in ctx.facts["design_review_findings"]


def test_design_review_gate_ignores_url_port_numbers(tmp_path):
    """A URL like https://example.com:443 must not be treated as a citation."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md",
                   _contract_design("See https://example.com:443/path and http://localhost:8080/api."))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"URL port numbers should not be flagged; got {result!r}"
    assert "443" not in ctx.facts["design_review_advisory"]
    assert "8080" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_accepts_bare_name_citation_to_real_line(tmp_path):
    """A citation like `somefile.py:2` resolves to a tracked file whose path ends with it."""
    source = tmp_path / "src"
    source.mkdir()
    sub = source / "deep" / "path"
    sub.mkdir(parents=True)
    (sub / "somefile.py").write_text("line1\nline2\nline3\n")
    # Simulate git ls-files by creating a .git directory so git works
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=str(source), check=True)
    subprocess.run(["git", "add", "."], cwd=str(source), check=True)
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(source), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    # Cite a bare name that resolves via git ls-files to deep/path/somefile.py
    ctx.write_text("design.md", _contract_design("See somefile.py:2 for the implementation."))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"bare-name citation should resolve; got {result!r}"
    assert "somefile.py" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_catches_backticked_citation_past_end(tmp_path):
    """A backticked `path/to/file.py:99` citation past the file's end is caught."""
    source = tmp_path / "src"
    source.mkdir()
    sub = source / "path" / "to"
    sub.mkdir(parents=True)
    (sub / "file.py").write_text("line1\nline2\nline3\n")
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(source), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md",
                   _contract_design("See `path/to/file.py:99` for details."))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    assert ops.gate_design_review(ctx) == "proceed"
    advisory = ctx.facts["design_review_advisory"]
    assert "99" in advisory and "3 lines" in advisory


def test_design_review_gate_skips_unresolvable_node_modules_citation(tmp_path):
    """A citation of node_modules/... that is not there is not a problem."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md",
                   _contract_design("See node_modules/lodash/index.js:42 for the implementation."))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"unresolvable node_modules citation should be skipped; got {result!r}"
    assert "node_modules" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_skips_ambiguous_bare_name_citation(tmp_path):
    """A bare name two tracked files share is not checked."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "a").mkdir()
    (source / "b").mkdir()
    (source / "a" / "index.ts").write_text("line1\n")
    (source / "b" / "index.ts").write_text("line1\n")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=str(source), check=True)
    subprocess.run(["git", "add", "."], cwd=str(source), check=True)
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(source), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    # Cite a bare name that matches two tracked files — should be skipped, not checked
    ctx.write_text("design.md", _contract_design("See index.ts:999 for the implementation."))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"ambiguous bare-name citation should be skipped; got {result!r}"
    assert "index.ts" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_ignores_code_quote_after_asks_to_change(tmp_path):
    """"the brief asks to change "do_thing()"" is not a finding."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md",
                   _contract_design('The brief asks to change "do_thing()" to use async.'))
    ctx.write_json("brief.json", {"name": "x", "brief": "refactor the module"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"code quote after 'asks to change' should not be a finding; got {result!r}"
    assert "do_thing" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_catches_misquote_after_brief_says_comma(tmp_path):
    """A misquote after "the brief says, " is still caught."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md",
                   _contract_design('The brief says, "enable fast mode" is required.'))
    ctx.write_json("brief.json", {"name": "x", "brief": "make it work correctly"})
    import refactory_dsh.design_ops as ops
    assert ops.gate_design_review(ctx) == "proceed"
    assert "enable fast mode" in ctx.facts["design_review_advisory"]


def test_design_review_gate_accepts_backticked_marker(tmp_path):
    """A site marked with backtick-wrapped ``[changed]`` is not flagged as unmarked."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "Everything is fine.\n\nVERDICT: PASS\n")
    design = (
        "# Do the thing\n\n"
        "## Summary\n\nFix it.\n\n"
        "## Failure trace\n\nN/A — this is a new feature.\n\n"
        "## Ask mapping\n\n"
        "| Ask (quoted from brief) | Design section | Test |\n"
        "|---|---|---|\n"
        '| "do a thing" | § Changes | test.py |\n\n'
        "## Rule and sites\n\n"
        "- `file.py:1` — `[changed]`\n\n"
        "## Docs\n\n- docs/models.md\n\n"
        "## Changes\n\n- file.py\n"
    )
    ctx.write_text("design.md", design)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"backticked marker should be accepted; got {result!r}"
    assert "marker" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_accepts_colon_description_marker(tmp_path):
    """A site marked with ``[changed: description]`` is not flagged as unmarked."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "Everything is fine.\n\nVERDICT: PASS\n")
    design = (
        "# Do the thing\n\n"
        "## Summary\n\nFix it.\n\n"
        "## Failure trace\n\nN/A — this is a new feature.\n\n"
        "## Ask mapping\n\n"
        "| Ask (quoted from brief) | Design section | Test |\n"
        "|---|---|---|\n"
        '| "do a thing" | § Changes | test.py |\n\n'
        "## Rule and sites\n\n"
        "- `file.py:1` — [changed: also reads command_history]\n\n"
        "## Docs\n\n- docs/models.md\n\n"
        "## Changes\n\n- file.py\n"
    )
    ctx.write_text("design.md", design)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", f"colon-description marker should be accepted; got {result!r}"
    assert "marker" not in ctx.facts["design_review_advisory"]


def test_design_review_gate_catches_misspelled_marker(tmp_path):
    """A site with ``[chnaged]`` gets the typo-detection message, as an advisory item."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "Everything is fine.\n\nVERDICT: PASS\n")
    design = (
        "# Do the thing\n\n"
        "## Summary\n\nFix it.\n\n"
        "## Ask mapping\n\n| Ask (quoted from brief) | Design section | Test |\n|---|---|---|\n"
        "| \"do a thing\" | § Changes | test_tool.py |\n\n"
        "## Rule and sites\n\n"
        "- `file.py:1` — [chnaged]\n\n"
        "## Docs\n\n- docs/models.md\n\n"
        "## Changes\n\n- file.py\n"
    )
    ctx.write_text("design.md", design)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    verdict = ops.gate_design_review(ctx)
    assert verdict == "proceed"
    advisory = ctx.facts["design_review_advisory"]
    assert "possible misspelled marker" in advisory
    assert "chnaged" in advisory


def test_design_review_gate_marked_site_with_bracket_words_no_problem(tmp_path):
    """A marked site whose line also contains ``[test]`` or a link raises no problem."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "Everything is fine.\n\nVERDICT: PASS\n")
    design = (
        "# Do the thing\n\n"
        "## Summary\n\nFix it.\n\n"
        "## Failure trace\n\nN/A — this is a new feature.\n\n"
        "## Ask mapping\n\n"
        "| Ask (quoted from brief) | Design section | Test |\n"
        "|---|---|---|\n"
        '| "do a thing" | § Changes | test.py |\n\n'
        "## Rule and sites\n\n"
        "- `file.py:1` — **[changed]** updates [test] and see [list](x.md)\n\n"
        "## Docs\n\n- docs/models.md\n\n"
        "## Changes\n\n- file.py\n"
    )
    ctx.write_text("design.md", design)
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    result = ops.gate_design_review(ctx)
    assert result == "proceed", (
        f"marked site with [test] and link should not raise problem; got {result!r}"
    )


def test_design_review_gate_reloop_findings_do_not_claim_pass(tmp_path):
    """A relooped design's design_review_findings no longer says nothing blocks it."""
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "design_review_attempts_used": 0})
    ctx.write_text("design.md", _CONTRACT_DESIGN.replace("## Changes\n\n- file.py\n", ""))
    ctx.write_json("brief.json", {"name": "x", "brief": "do a thing"})
    import refactory_dsh.design_ops as ops
    verdict = ops.gate_design_review(ctx)
    assert verdict == "reloop"
    assert "nothing blocks" not in ctx.facts["design_review_findings"]


def test_a_design_run_produces_a_checked_artifact():
    """End to end, with a fake model: design, emit, and the check passing."""
    mode = DesignMode()
    state = mode.state_dict()
    mode.load_state_dict(state)
    workdir = Path(".factory/design-run")
    shutil.rmtree(workdir, ignore_errors=True)
    model = FakeModel(_design_answers([render({"tool.py": "print('built')\n"})]))
    result = mode.run(
        workdir,
        model=model,
        task={"name": ARTIFACT, "brief": "a tool", "check": "python tool.py"},
        verdicts={"gate_strategy": "proceed"},
    )
    assert result.exit_ok, result.error
    payload = json.loads((workdir / "design.json").read_text())
    assert payload["name"] == ARTIFACT
    assert payload["checked"] is True
    # With no target, the run started a new project and worked in its own worktree
    # on its own branch (#351): the artifact is that worktree, committed there.
    assert payload["branch"].startswith("refactory/run-")
    assert Path(payload["artifact"], "tool.py").read_text() == "print('built')\n"
    assert payload["head"] != payload["base"], "the attempt is a commit on the run's branch"
    assert not (ARTIFACTS / ARTIFACT).exists(), "nothing is written under .factory/artifacts"


def test_a_design_that_never_passes_its_check_is_delivered_as_a_check_failing_draft():
    """Its attempts applied, so at the check limit the run is not a failure but a draft
    marked check failing (b45a): loud in its title and its description, not in its exit."""
    mode = DesignMode()
    workdir = Path(".factory/design-fail")
    shutil.rmtree(workdir, ignore_errors=True)
    # Each emission must be distinct so materialise does not refuse it as "changed nothing".
    emits = [render({"tool.py": f"raise SystemExit({i})\n"}) for i in range(1, 5)]
    model = FakeModel(_design_answers(emits))
    result = mode.run(
        workdir,
        model=model,
        task={"name": ARTIFACT, "brief": "a tool", "check": "python tool.py"},
        verdicts={"gate_strategy": "proceed"},
    )
    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert fact["reason"] == "check attempts exhausted after 3 check failure(s)"
    assert fact["check_failing"] is True


def test_a_revision_records_the_state_it_came_from(tmp_path):
    """Provenance is the state, not only the name of a class that may have moved.

    A mode is not fixed: the outer loop edits it. "Revised from
    `circle_packing/adaptive`" therefore names a class that may have changed since,
    and a revision recorded that way cannot be reproduced or diffed against what it
    actually started from.
    """
    result = edit_module.edit(
        _edit_config(name=NAME, mode="circle_packing/adaptive",
                   brief="restart on a stall", workdir=tmp_path / "run"),
        client=FakeModel(_answers()),
    )
    assert result["revision_of"] == "circle_packing/adaptive"
    assert result["revision_of_digest"], "no digest of the state that was revised"

    recorded = json.loads(
        (registry.workspace_root(Path(".")) / NAME / registry.ORIGIN).read_text()
    )
    state = recorded["revision_of_state"]
    assert sorted(state) == sorted(
        registry.resolve(Path("."), "circle_packing/adaptive")[1]().state_dict()
    )
    assert recorded["revision_of_digest"] == result["revision_of_digest"]

    # The digest identifies the state it was recorded from.
    from refactory_dsh.optim.base import candidate_id

    assert recorded["revision_of_digest"] == candidate_id(state)


# ── the comparison, on the path a person actually runs ─────────────────────
#
# `refactory ceo "…" --mode edit` lowers `default_pack/edit` and runs its nodes on
# DSH; it does not call `edit.edit()`. The comparison therefore belongs to the
# `smoke` node, which is the step that has just scored the new mode, and these hold
# it there. Before this existed the CLI recorded `revision_of` and nothing else: no
# before score, no after score, no refusal — the documented behaviour of `--mode
# edit`, implemented only in a function the CLI never reached.

#: The declarations lowering writes for the smoke node of an edit run.
_SMOKE_DECLARATIONS = {
    "smoke": {
        "reads": [],
        "writes": ["smoke.json", "revision.json"],
        "produces": ["smoke_score", "smoke_ok", "before_score", "after_score",
                     "regressed", "revision_note"],
    },
}


def _smoke_run(tmp_path, monkeypatch, *, scores, revision="counting_pack/fixed_step",
               package_path=None, allow=False):
    """A run directory the smoke node can be invoked in, as the runtime invokes it.

    `counting_pack` is a shipped, deterministic package, so the node's own smoke of
    the new mode is real; only the two scores the comparison reads are stubbed, so
    what is under test is the comparison rather than a search for a mode whose
    numbers happen to differ.
    """
    from refactory_dsh import lowering, sandbox
    from refactory_dsh.sandbox import SandboxResult

    workdir = tmp_path / "run"
    (workdir / ".refactory").mkdir(parents=True)
    (workdir / ".refactory" / "declarations.json").write_text(
        json.dumps(_SMOKE_DECLARATIONS)
    )
    # A shipped, deterministic package, so the node under test does not depend on
    # anything this machine happened to have created.
    facts = {"name": "circle_packing", "revision_of": revision}
    if package_path is not None:
        facts["package_path"] = str(package_path)
    (workdir / "facts.json").write_text(json.dumps(facts))

    def stub(identifier, workdir_, *, project=None, model=False, **kwargs):
        return SandboxResult(ok=True, payload={"score": scores[identifier], "iterations": 2})

    monkeypatch.setattr(sandbox, "smoke", stub)
    if allow:
        monkeypatch.setenv(lowering.KNOBS_ENV, json.dumps({"allow_regression": True}))
    return workdir


def test_the_smoke_node_refuses_a_revision_that_scores_lower(tmp_path, monkeypatch):
    """A regression is refused on the path the CLI takes, and the run says both scores."""
    from refactory_dsh.run import run_node

    workdir = _smoke_run(tmp_path, monkeypatch, scores={
        "circle_packing/adaptive": 0.4,
        "counting_pack/fixed_step": 0.9,
    })
    with pytest.raises(ValueError, match="scores 0.4 where counting_pack/fixed_step scores 0.9"):
        run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    # The evidence outlives the refusal: a person reading the run can see both numbers.
    recorded = json.loads((workdir / "revision.json").read_text())
    assert recorded["before_score"] == 0.9
    assert recorded["after_score"] == 0.4
    assert recorded["regressed"] is True


def test_the_smoke_node_accepts_the_same_revision_when_the_trade_is_intended(
    tmp_path, monkeypatch
):
    """`allow_regression=1` is the person saying the score is not the whole story."""
    from refactory_dsh.run import run_node

    package = tmp_path / "counting_pack"
    package.mkdir()
    workdir = _smoke_run(
        tmp_path, monkeypatch, allow=True,
        package_path=package,
        scores={"circle_packing/adaptive": 0.4, "counting_pack/fixed_step": 0.9},
    )
    produced = run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    assert produced["regressed"] is None
    assert (produced["before_score"], produced["after_score"]) == (None, 0.4)
    assert "intentionally skipped" in produced["revision_note"]
    recorded = json.loads((package / ".origin.json").read_text())
    assert recorded["revision_of"] == "counting_pack/fixed_step"
    assert recorded["before_score"] is None and recorded["after_score"] == 0.4
    assert recorded["revision_of_digest"], "the provenance names no state"


def test_an_improvement_is_not_a_regression(tmp_path, monkeypatch):
    from refactory_dsh.run import run_node

    workdir = _smoke_run(tmp_path, monkeypatch, scores={
        "circle_packing/adaptive": 1.0,
        "counting_pack/fixed_step": 0.9,
    })
    produced = run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    assert produced["regressed"] is False
    assert produced["revision_note"] == "compared with counting_pack/fixed_step: 0.9 → 1.0"


def test_a_comparison_names_the_defaults_the_revision_changed(tmp_path, monkeypatch):
    """dm-263: an edit raised `budget`'s default from 3 to 8 and reported 0.875 → 1.08 as
    if both ran alike. The comparison says so beside the numbers; it does not re-score."""
    from refactory_dsh import lowering, sandbox
    from refactory_dsh.run import run_node
    from refactory_dsh.sandbox import SandboxResult

    source = registry.resolve(Path("."), "circle_packing/adaptive")[1]().state_dict()["knobs"]
    revised = {name: entry["value"] for name, entry in source.items()}
    assert revised["budget"] == 3
    revised["budget"] = 8
    calls = []

    def stub(identifier, workdir_, *, project=None, model=False, knobs=None, **kwargs):
        calls.append(knobs)
        if Path(workdir_).name == "smoke-source":
            return SandboxResult(ok=True, payload={"score": 0.875, "iterations": 2})
        ran = {**revised, **({"budget": int(knobs["budget"])} if knobs else {})}
        return SandboxResult(ok=True, payload={"score": 1.08, "iterations": 2, "knobs": ran})

    workdir = _smoke_run(tmp_path, monkeypatch, revision="circle_packing/adaptive",
                         scores={})
    monkeypatch.setattr(sandbox, "smoke", stub)
    produced = run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    assert produced["revision_note"] == (
        "compared with circle_packing/adaptive: 0.875 → 1.08 (not like for like: budget 3 → 8)")
    assert produced["before_score"] == 0.875 and produced["regressed"] is False
    assert json.loads((workdir / "revision.json").read_text())["changed_defaults"] == ["budget 3 → 8"]
    assert len(calls) == 2, "said, not re-scored"

    # A knob both smokes were given the same value of is like for like.
    monkeypatch.setenv(lowering.KNOBS_ENV, json.dumps({"sample_knobs": '["budget=8"]'}))
    produced = run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    assert produced["revision_note"] == "compared with circle_packing/adaptive: 0.875 → 1.08"


def test_the_sandboxed_smoke_reports_the_knob_values_it_ran_with(tmp_path):
    """What the comparison above reads: the defaults, with the sample knobs over them."""
    from refactory_dsh import sandbox

    result = sandbox.smoke("circle_packing/adaptive", tmp_path / "smoke", project=Path("."),
                           knobs={"budget": "5"})
    assert result.ok, result.problem()
    defaults = registry.resolve(Path("."), "circle_packing/adaptive")[1]().state_dict()["knobs"]
    assert result.payload["knobs"] == {
        **{name: entry["value"] for name, entry in defaults.items()}, "budget": 5}


def test_a_mode_that_cannot_report_its_knobs_still_passes_its_smoke(tmp_path):
    """The knob report runs after the mode has run; it cannot turn a pass into a failure."""
    from refactory_dsh import sandbox

    root = registry.workspace_root(tmp_path) / "stateless_pkg"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n"
        "from refactory_packages.circle_packing.mode import CirclePacking\n\n\n"
        "class Stateless(CirclePacking):\n"
        "    ran = False\n\n"
        "    def evaluate(self, *args, **kwargs):\n"
        "        result = super().evaluate(*args, **kwargs)\n"
        "        type(self).ran = True\n"
        "        return result\n\n"
        "    def state_dict(self):\n"
        "        if type(self).ran:\n"
        "            raise RuntimeError('no state after the run')\n"
        "        return super().state_dict()\n\n\n"
        "PACKAGE = Package(name='stateless_pkg', description='x', modes=(Stateless,),\n"
        "                  ops=('refactory_packages.circle_packing.ops',))\n")
    result = sandbox.smoke("stateless_pkg/adaptive", tmp_path / "smoke", project=tmp_path)
    assert result.ok, result.problem()
    assert result.payload["exit_ok"] is True and result.payload["knobs"] is None


def test_a_revision_records_its_passing_smoke_in_its_provenance(tmp_path, monkeypatch):
    """One write: the revision's provenance and the pass together; a refused one stays pending."""
    from refactory_dsh.run import run_node

    package = tmp_path / "revised"
    package.mkdir()
    origin = package / registry.ORIGIN
    origin.write_text(json.dumps({"origin": "create-mode", "smoke": "pending"}))
    workdir = _smoke_run(tmp_path, monkeypatch, package_path=package, scores={
        "circle_packing/adaptive": 1.0, "counting_pack/fixed_step": 0.9})
    run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    recorded = json.loads(origin.read_text())
    assert (recorded["smoke"], recorded["revision_of"]) == ("passed", "counting_pack/fixed_step")

    origin.write_text(json.dumps({"origin": "create-mode", "smoke": "pending"}))
    regressed = _smoke_run(tmp_path / "again", monkeypatch, package_path=package, scores={
        "circle_packing/adaptive": 0.4, "counting_pack/fixed_step": 0.9})
    with pytest.raises(ValueError, match="not an improvement"):
        run_node("refactory_dsh.create_ops:smoke", workdir=regressed, node_id="smoke")
    assert json.loads(origin.read_text())["smoke"] == "pending"


def test_a_revision_of_an_unscored_mode_is_recorded_as_uncompared(tmp_path, monkeypatch):
    """There is no number to compare against, and saying so beats inventing one."""
    from refactory_dsh.run import run_node

    workdir = _smoke_run(
        tmp_path, monkeypatch, revision="default_pack/design",
        scores={"circle_packing/adaptive": 0.4},
    )
    produced = run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    assert produced["before_score"] is None
    assert produced["regressed"] is False
    assert "unscored" in produced["revision_note"]


def test_a_revision_of_a_mode_that_calls_a_model_is_not_refused_on_one_sample(
    tmp_path, monkeypatch
):
    """One smoke score of a model call is a sample, not a measurement.

    Refusing a revision on one sample would refuse the wrong thing, so a revision of a
    mode that calls a model is recorded as not compared rather than as passing.
    """
    from refactory_dsh import registry
    from refactory_dsh.run import run_node

    source = registry.workspace_root(Path(".")) / "agent_probe"
    shutil.rmtree(source, ignore_errors=True)
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        "from refactory_dsh.nn import Package\n"
        "from .mode import Drafting\n\n"
        "PACKAGE = Package(name='agent_probe', description='drafts with an agent',\n"
        "                  modes=(Drafting,), ops=('refactory_packages.agent_probe.ops',))\n"
    )
    (source / "ops.py").write_text(
        "def finish(ctx):\n"
        "    ctx.write_json('out.json', {'done': True})\n"
        "    return {'done': True}\n"
    )
    (source / "mode.py").write_text(
        "from refactory_dsh import nn\n\n"
        "OPS = 'refactory_packages.agent_probe.ops'\n\n\n"
        "class Drafting(nn.Mode):\n"
        "    name = 'draft'\n"
        "    description = 'one agent writes a line'\n\n"
        "    def build(self):\n"
        "        graph = nn.Graph()\n"
        "        graph.add(nn.Node('draft', 'agent', prompt_template='write a line',\n"
        "                          writes=('line.txt',), produces=('line',)))\n"
        "        graph.add(nn.Node('finish', 'op', call=f'{OPS}:finish',\n"
        "                          reads=('line.txt',), writes=('out.json',),\n"
        "                          produces=('done',)))\n"
        "        graph.connect('draft', 'finish')\n"
        "        return graph\n\n"
        "    def score(self, run):\n"
        "        return 1.0 if run.facts.get('done') else 0.0\n"
    )
    try:
        workdir = _smoke_run(
            tmp_path, monkeypatch, revision="agent_probe/draft",
            scores={"circle_packing/adaptive": 0.4},
        )
        produced = run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
        assert produced["before_score"] is None
        assert "calls a model" in produced["revision_note"]
    finally:
        shutil.rmtree(source, ignore_errors=True)


def test_the_run_summary_reports_what_a_revision_recorded(tmp_path):
    """`ceo`/`run` end with what the run recorded, not only DSH's final word."""
    from refactory_dsh.interactive import run_summary

    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / "facts.json").write_text(json.dumps({
        "name": "counting_revised",
        "revision_of": "counting_pack/fixed_step",
        "before_score": 0.5,
        "after_score": 1.0,
        "regressed": False,
        "package_path": "/tmp/nowhere",
    }))
    line = run_summary(workdir)
    assert line.startswith("refactory: what the run recorded")
    for wanted in ("revision_of", "before_score", "after_score", "regressed"):
        assert wanted in line


def test_the_run_summary_says_nothing_when_there_is_nothing_to_say(tmp_path):
    from refactory_dsh.interactive import run_summary

    workdir = tmp_path / "run"
    workdir.mkdir()
    assert run_summary(workdir) == ""
    (workdir / "facts.json").write_text("{ not json")
    assert run_summary(workdir) == ""


def test_a_comparison_with_no_package_path_writes_no_provenance(tmp_path, monkeypatch):
    """The provenance belongs to the package; a node with no package writes none.

    `Path("") / ".origin.json"` is `./.origin.json`, so an unguarded write put a
    revision's provenance into whatever directory the node was standing in — which for
    a hand-invoked node is the checkout. Found exactly that way: a run of the suite
    left an untracked `.origin.json` at the repository root.
    """
    from refactory_dsh.run import run_node

    stray = Path(".origin.json")
    stray.unlink(missing_ok=True)
    workdir = _smoke_run(
        tmp_path, monkeypatch, revision="default_pack/design",
        scores={"circle_packing/adaptive": 0.4},
    )
    run_node("refactory_dsh.create_ops:smoke", workdir=workdir, node_id="smoke")
    assert not stray.exists(), "the comparison wrote provenance into the working directory"
