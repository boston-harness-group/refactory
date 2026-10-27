"""Design mode's judges: the reviews report, and the QA judge and the overwatch decide.

Each judge writes a JSON verdict (`refactory_dsh.judges`) and its gate reads that file
with ``json.loads``; nothing in design mode reads a decision out of a review's prose.
The check is decided first and stays binding. The cases are the runs that failed on the
parser and the audit: b37b and measure4-dm149, whose reviewers quoted the blocking
marker's format and were sent back for it, and b39b, whose correction audit refused a
change the overwatch had passed. Everything here runs through the real design graph on
the walker, with a scripted model, except the last tests, which run a gate the way the
spine does: as its own process, through ``refactory_dsh.run``.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from refactory_dsh import authoring_evidence as evidence
from refactory_dsh import create_ops, delivery, design_ops, judges, lowering, nn
from refactory_dsh.create import CreateMode
from refactory_dsh.design import DesignMode
from refactory_dsh.edit import EditMode
from refactory_dsh.run import FAILURE_MARKER, main as runner_main
from refactory_dsh.steering import Ledger
from tests_support import (DESIGN_DOC_ANSWER, OVERWATCH_JUDGE_MARK,
                           PITFALLS_ANSWER, QA_JUDGE_MARK, RESEARCH_ANSWER, VERIFICATION_ANSWER,
                           FakeModel, Scripted, git, git_repo, overwatch_judgment, qa_judgment,
                           render)

NAME = "judged_tool"
EMIT_MARK = "Write the artifact `"

#: Passing reviews, in the order the review fork asks for them, then the verifier's.
HEALTH, CODE, QA = "health: ran it, it prints built", "code review: matches the design", "qa: ok"


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def _round(emission: str, *, reviews: tuple[str, str, str] = (HEALTH, CODE, QA),
           verifier: str = VERIFICATION_ANSWER, judge: str | list[str] | None = None,
           overwatch: str | list[str] | None = None) -> list[str]:
    """One attempt's answers: the emission, the three reviews and the verifier beside the
    check, then the judges (each left to its default when not given, and given a list
    when it is asked again)."""
    def answers(value: str | list[str] | None) -> list[str]:
        return [] if value is None else [value] if isinstance(value, str) else list(value)

    return [emission, *reviews, verifier, *answers(judge), *answers(overwatch)]


def _answers(*rounds: list[str]) -> list[str]:
    head = [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER]
    return head + [answer for one in rounds for answer in one]


def _design(run: Path, model: FakeModel, repo: Path, *, check: str = "python tool.py",
            **knobs: int) -> nn.RunResult:
    mode = DesignMode()
    state = mode.state_dict()
    for key, value in knobs.items():
        state["knobs"][key]["value"] = value
    mode.load_state_dict(state)
    return mode.run(run, model=model, verdicts={"gate_strategy": "proceed"},
                    task={"name": NAME, "brief": "a tool that prints built", "check": check,
                          "target": str(repo)})


def _prompts(model: FakeModel, mark: str) -> list[int]:
    """The positions, in the order the model was asked, of the prompts carrying ``mark``."""
    return [index for index, prompt in enumerate(model.prompts) if mark in prompt]


def _tool(word: str, code: int = 0) -> str:
    return render({"tool.py": f"print({word!r})\nraise SystemExit({code})\n"})


# ── b37b and measure4-dm149: prose that quotes the marker is not a finding ────
#: Lines from b37b's own reviews, which the parser read as blocking findings (a heading
#: naming the format, and a sentence that wraps onto a line starting with the marker),
#: and a tally line of the kind measure4-dm149's reviewers wrote.
QUOTING_HEALTH = (
    "# Health check\n\n"
    "## 11. BLOCKING: <path>: format preserved in all prompts\n\n"
    "**Check:** every reviewer prompt still contains the format.\n\n**Result:** PASS\n")
QUOTING_CODE = (
    "# Code review\n\n"
    "Ran the parser over each marker form:\n"
    "- `BLOCKING:` → 1 finding ✓\n"
    "- `**BLOCKING:**` → 1 finding ✓\n\n"
    "Nothing in the change is defective.\n")
QUOTING_QA = (
    "# QA\n\n**Adversarial check:** scanned every node for any node that mentions\n"
    "`BLOCKING:` but does NOT contain `MARKER_RULE`. Found zero such nodes.\n\n"
    "**Result:** PASS.\n")
QUOTING_VERIFIER = "BLOCKING: tool.py:1: prints a word the verifier did not expect\n"


def test_prose_quoting_the_marker_does_not_send_the_change_back(tmp_path):
    """b37b: reviews that quote the marker's format, and a verifier with a marked line,
    with a QA judge that proceeds: one emission, no reloop, the judge's dismissals on
    record. The parser the QA gate used before would have sent the same reviews back."""
    repo = git_repo(tmp_path / "repo")
    judge = qa_judgment("proceed", dismissed=[
        {"source": "health_checker", "finding": "## 11. BLOCKING: <path>: ...",
         "why": "a heading naming the format, not a finding"}])
    model = FakeModel(_answers(_round(_tool("built"),
                                      reviews=(QUOTING_HEALTH, QUOTING_CODE, QUOTING_QA),
                                      verifier=QUOTING_VERIFIER, judge=judge,
                                      overwatch=overwatch_judgment("pass"))))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    assert len(_prompts(model, EMIT_MARK)) == 1
    assert not result.facts.get("review_attempts_used")
    gate = (run / "reviews/qa-gate.txt").read_text()
    assert gate.startswith("proceed: the QA judge found nothing that must be fixed"), gate
    assert "dismissed [health_checker] ## 11. BLOCKING: <path>: ..." in gate
    # The verifier went to the judge whole and was not parsed for findings.
    assert QUOTING_VERIFIER.strip() in (run / "reviews/qa-synthesized.md").read_text()
    assert not (run / "reviews/verification-unresolved.md").exists()
    # The same text through the parser the QA gate used: core findings, so a reloop.
    for text in (QUOTING_HEALTH, QUOTING_CODE, QUOTING_QA):
        core, _ = create_ops.split_core(create_ops.blocking_findings(text))
        assert core, text


def test_a_judge_redo_sends_its_must_fix_to_the_builder_once(tmp_path):
    """The QA judge's redo goes back to emit with its must-fix item as the feedback, and
    spends one review attempt; the next attempt proceeds."""
    repo = git_repo(tmp_path / "repo")
    must_fix = {"source": "code_reviewer", "where": "tool.py:1", "what": "prints the wrong word",
                "why": "the brief asks for built"}
    model = FakeModel(_answers(
        _round(_tool("bilt"), judge=qa_judgment("redo", must_fix=[must_fix],
                                                follow_ups=["add a --quiet flag"])),
        _round(_tool("built"), overwatch=overwatch_judgment("pass"))))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    emits = _prompts(model, EMIT_MARK)
    assert len(emits) == 2, "exactly one reloop"
    assert len(_prompts(model, QA_JUDGE_MARK)) == 2
    second = model.prompts[emits[1]]
    assert "[code_reviewer] tool.py:1: prints the wrong word (why: the brief asks for built)" in second
    assert "add a --quiet flag" in second, "follow-ups ride along, to fix only if cheap"
    assert result.facts["review_attempts_used"] == 1
    assert (run / "reviews/qa-gate.txt").read_text().startswith("proceed:")


def test_the_judges_see_what_sent_the_attempt_back(tmp_path):
    """Both judges get `review_feedback`, what sent this attempt back, as the reviewers
    beside the check do. By the time a judge runs, `validation` holds only this round's
    check and QA results, so a judge given that could not tell whether the attempt fixed
    what an earlier round named."""
    repo = git_repo(tmp_path / "repo")
    must_fix = {"source": "code_reviewer", "where": "tool.py:1", "what": "prints the wrong word",
                "why": "the brief asks for built"}
    model = FakeModel(_answers(
        _round(_tool("bilt"), judge=qa_judgment("redo", must_fix=[must_fix])),
        _round(_tool("built"), overwatch=overwatch_judgment("pass"))))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    first, second = (model.prompts[index] for index in _prompts(model, QA_JUDGE_MARK))
    (overwatch,) = (model.prompts[index] for index in _prompts(model, OVERWATCH_JUDGE_MARK))
    sent_back = "[code_reviewer] tool.py:1: prints the wrong word (why: the brief asks for built)"
    assert sent_back not in first
    assert f"## What sent this attempt back\n{result.facts['review_feedback']}" in second
    assert sent_back in second
    assert f"What sent this attempt back: {result.facts['review_feedback']}" in overwatch
    assert sent_back in overwatch


def test_a_failing_check_goes_back_before_any_judge(tmp_path):
    """The check stays binding: an attempt whose check fails goes back to emit from
    gate_check, and no judge is asked about it. The judge sees only the passing attempt."""
    repo = git_repo(tmp_path / "repo")
    # The first attempt's reviews are answered by default beside its failing check: the
    # script's next answer is the second emission.
    model = FakeModel(_answers([_tool("built", 1)],
                               _round(_tool("built"), overwatch=overwatch_judgment("pass"))))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    emits = _prompts(model, EMIT_MARK)
    judged = _prompts(model, QA_JUDGE_MARK)
    assert len(emits) == 2
    assert len(judged) == 1 and judged[0] > emits[1], "the judge ran only on the passing attempt"
    assert len(_prompts(model, OVERWATCH_JUDGE_MARK)) == 1
    assert result.facts.get("check_attempts_used") == 1
    assert not result.facts.get("review_attempts_used")


# ── b39b: the correction audit is the overwatch's evidence, not a verdict ─────
CALC = "def validate(x):\n    return x > 0\n\n\ndef legacy():\n    return 1\n"
CALC_TEST = "from calc import validate\n\n\ndef test_unrelated():\n    assert validate(5) is True\n"
CORRECTION = "make zero valid, drop legacy(), and leave app.py alone"
#: Sites the evidence map names for the correction: app.py, which the change leaves as
#: the base has it (as b39b's dropped guard was left); the comparison in calc.py, whose
#: cited test passes with the change reverted; and legacy(), which the change deletes.
SITES = [{"site": "app.py:1", "tests": ["test_calc.py::test_unrelated"], "scope": "local"},
         {"site": "calc.py:2", "tests": ["test_calc.py::test_unrelated"], "scope": "local"},
         {"site": "calc.py:5", "tests": ["test_calc.py::test_unrelated"], "scope": "local"}]


def _calc_repo(tmp_path: Path) -> Path:
    repo = git_repo(tmp_path / "repo")
    (repo / "calc.py").write_text(CALC)
    (repo / "test_calc.py").write_text(CALC_TEST)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "calc")
    return repo


def _correction_run(tmp_path: Path, correction: str) -> tuple[Path, str]:
    """A run directory whose ledger holds one correction from the person, and its id."""
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, "a calc whose validate accepts zero").note("correction", correction)
    [key] = [f"ledger-{index + 1}" for index, entry in enumerate(evidence.intent(run))
             if entry["kind"] == "correction"]
    return run, key


def test_b39b_removed_unchanged_and_untested_sites_with_a_passing_overwatch_reach_the_report(tmp_path):
    """b39b: the audit reports a site the change left as the base has it as unchanged --
    measured as not met, for the overwatch to decide -- a deleted site as removed, and a
    site no cited test reaches. The overwatch reads that, judges the correction met with
    evidence, passes, and the run reaches its report and delivers. The pull request says
    what the overwatch judged. The audit the overwatch gate ran before would have refused
    the same evidence."""
    repo = _calc_repo(tmp_path)
    run, key = _correction_run(tmp_path, CORRECTION)
    # Evidence with a line break and a heading of its own, which the PR must not render.
    verdict = overwatch_judgment("pass", corrections=[
        {"id": key, "met": True,
         "evidence": "calc.py:2 returns x >= 0;\n## Injected heading\napp.py is unchanged"}])
    emission = render({"calc.py": "def validate(x):\n    return x >= 0\n"})
    model = FakeModel(_answers(_round(emission, overwatch=verdict)))
    model._correction_sites = {CORRECTION: SITES}
    result = _design(run, model, repo,
                     check="python -c 'from calc import validate; assert validate(0)'")

    assert result.exit_ok, result.error
    [item] = json.loads((run / "correction-audit.json").read_text())
    sites = {site["site"]: site for clause in item["clauses"] for site in clause["sites"]}
    assert sites["app.py:1"]["status"] == "unchanged"
    assert sites["calc.py:5"]["status"] == "removed" and sites["calc.py:5"]["removed"] == "deleted"
    assert sites["calc.py:2"]["status"] == "fail"
    assert [t["revert_result"] for t in sites["calc.py:2"]["tests"]] == ["pass"]
    # Measured as not met: never a pass, whatever the overwatch then decides.
    assert item["status"] == "fail" and [c["status"] for c in item["clauses"]] == ["fail"]
    # The overwatch read the audit, every finding in it, before judging.
    [asked] = [model.prompts[i] for i in _prompts(model, OVERWATCH_JUDGE_MARK)]
    assert "app.py:1: unchanged" in asked and "calc.py:5: removed (deleted)" in asked
    assert f"{key} clause 1: site app.py:1 is unchanged" in asked
    assert "site calc.py:2 not reached by any test" in asked
    # It passed, the report ran, and nothing was sent back.
    assert (run / "design.json").exists()
    assert (run / "reviews/overwatch-gate.txt").read_text().startswith("proceed: overwatch passed")
    assert not result.facts.get("review_attempts_used")
    follow_ups = (run / "reviews/follow-ups.md").read_text()
    assert "did not judge" not in follow_ups
    # The pull request takes its wording from the overwatch's verdict, not the audit's.
    body = (run / delivery.BODY).read_text()
    assert "- **Corrections, as the overwatch judged them:**" in body
    assert f"  - {key}: met — calc.py:2 returns x >= 0; ## Injected heading app.py is unchanged\n" in body
    assert "\n## Injected heading" not in body
    assert "Correction-audit checks to confirm by hand" not in body
    assert "site app.py:1 is unchanged" not in body and "site calc.py:5 is removed" not in body
    assert "and the overwatch audit passed" in body, body
    # The audit as the overwatch gate ran it refuses the same evidence.
    ctx = nn.RunContext(run, facts=result.facts, enforce_writes=False)
    old = evidence.correction_audit_problems(ctx)
    assert any("site app.py:1 is not in the diff" in problem for problem in old), old
    assert any("site calc.py:2 not reached by any test" in problem for problem in old), old


ZERO_TEST = "from calc import validate\n\n\ndef test_zero():\n    assert validate(0) is True\n"


def test_an_unapplied_correction_is_not_met_and_a_pass_that_says_so_goes_back(tmp_path):
    """D2 and D4: a correction the change never applied leaves its site unchanged, and the
    audit measures that as not met. An overwatch that marks the correction not met sends
    the change back even with a pass, naming the correction and its evidence, and spends a
    review attempt; the next round applies it and passes."""
    repo = git_repo(tmp_path / "repo")
    (repo / "calc.py").write_text("def validate(x):\n    return x > 0\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "calc")
    run, key = _correction_run(tmp_path, "make zero valid in calc.validate")
    unmet = overwatch_judgment("pass", corrections=[
        {"id": key, "met": False, "evidence": "calc.py:2 still returns x > 0"}])
    met = overwatch_judgment("pass", corrections=[{"id": key, "met": True, "evidence": "calc.py:2"}])
    model = FakeModel(_answers(
        _round(render({"README.md": "calc\n"}), overwatch=unmet),
        _round(render({"calc.py": "def validate(x):\n    return x >= 0\n",
                       "test_calc.py": ZERO_TEST}), overwatch=met)))
    model._correction_sites = {"make zero valid in calc.validate": [
        {"site": "calc.py:2", "tests": ["test_calc.py::test_zero"], "scope": "local"}]}
    result = _design(run, model, repo, check="true")

    assert result.exit_ok, result.error
    emits = _prompts(model, EMIT_MARK)
    assert len(emits) == 2 and result.facts["review_attempts_used"] == 1
    assert f"- correction {key}: NOT met — calc.py:2 still returns x > 0" in model.prompts[emits[1]]
    first, second = [model.prompts[i] for i in _prompts(model, OVERWATCH_JUDGE_MARK)]
    assert f"{key} clause 1: site calc.py:2 is unchanged" in first
    assert "What the audit measured as not met" in first
    assert "site calc.py:2 is unchanged" not in second
    assert (run / "reviews/overwatch-gate.txt").read_text().startswith("proceed: overwatch passed")


def test_a_site_outside_the_change_is_removed_only_when_the_change_deletes_it(tmp_path):
    """R07a, R07b: a deleted site is removed and an untouched one unchanged; a line past
    the end of its file, or a file that does not exist, names no code at all."""
    from refactory_dsh import base_check

    repo = _calc_repo(tmp_path)
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "-b", "run-branch")
    (repo / "calc.py").write_text("def validate(x):\n    return x >= 0\n")
    git(repo, "commit", "-q", "-am", "change")
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "worktree.json").write_text(json.dumps({
        "path": str(repo), "base": base, "head": git(repo, "rev-parse", "HEAD"),
        "branch": "run-branch", "target": str(repo), "run": str(run)}))
    ctx = nn.RunContext(run)
    assert base_check.site_outside_change(ctx, "calc.py:5") == "deleted"
    assert base_check.site_outside_change(ctx, "app.py:1") == "unchanged"
    assert base_check.site_outside_change(ctx, "app.py:9") == ""
    assert base_check.site_outside_change(ctx, "missing.py:1") == ""


def test_an_overwatch_redo_counts_toward_the_review_limit_and_stops_at_it(tmp_path):
    """An overwatch redo goes back to emit with its must-fix item and spends a review
    attempt; at the review limit the build loop stops, and the applied attempt goes to the
    report as a draft listing the item, with both attempts that named it (b45a)."""
    repo = git_repo(tmp_path / "repo")
    redo = overwatch_judgment("redo", must_fix=[
        {"source": "overwatch", "where": "tool.py:1", "what": "no test shows built is printed",
         "why": "the brief's only ask has no evidence"}])
    model = FakeModel(_answers(_round(_tool("built"), overwatch=redo),
                               _round(_tool("built!"), overwatch=redo)))
    result = _design(tmp_path / "run", model, repo, review_attempts=2)

    assert result.exit_ok, result.error
    fact = result.facts["draft_delivery"]
    assert fact["reason"] == "review reloop limit reached after 2 review cycle(s)"
    [item] = fact["unresolved"]
    assert "tool.py:1: no test shows built is printed" in item["text"]
    assert (item["source"], item["rounds"]) == ("overwatch", [1, 2])
    emits = _prompts(model, EMIT_MARK)
    assert len(emits) == 2
    assert "tool.py:1: no test shows built is printed" in model.prompts[emits[1]]


def test_a_judge_whose_verdict_cannot_be_read_is_asked_again_with_no_builder_round(tmp_path):
    """D3: each judge's unreadable verdict is read again once, judge only: the judge is
    asked again with the reason in its prompt, and no builder round or review attempt is
    spent. Both readable second answers decide the run."""
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(_answers(_round(
        _tool("built"), judge=[Scripted("Looks fine to me."), qa_judgment("proceed")],
        overwatch=[Scripted("PASS, everything holds"), overwatch_judgment("pass")])))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    assert len(_prompts(model, EMIT_MARK)) == 1
    assert not result.facts.get("review_attempts_used")
    qa_first, qa_second = [model.prompts[i] for i in _prompts(model, QA_JUDGE_MARK)]
    assert "Your last verdict could not be read" not in qa_first
    assert ("## Your last verdict could not be read\nqa-judge.json holds no JSON object"
            in qa_second)
    ow_first, ow_second = [model.prompts[i] for i in _prompts(model, OVERWATCH_JUDGE_MARK)]
    assert "Your last verdict could not be read" not in ow_first
    assert "overwatch-verdict.json holds no JSON object" in ow_second
    assert (run / "reviews/qa-gate.txt").read_text().startswith(
        "proceed: the QA judge found nothing that must be fixed")
    assert (run / "reviews/overwatch-gate.txt").read_text().startswith("proceed: overwatch passed")


def test_each_round_gets_its_own_reread(tmp_path):
    """The reread count is per round: round 1 spends its reread on each judge, and round 2,
    after the overwatch sends the change back, gets its own."""
    repo = git_repo(tmp_path / "repo")
    redo = overwatch_judgment("redo", must_fix=[{"what": "prints the wrong word", "severity": "high"}])
    model = FakeModel(_answers(
        _round(_tool("bilt"), judge=[Scripted("unreadable"), qa_judgment("proceed")],
               overwatch=[Scripted("unreadable"), redo]),
        _round(_tool("built"), judge=[Scripted("unreadable"), qa_judgment("proceed")],
               overwatch=[Scripted("unreadable"), overwatch_judgment("pass")])))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    assert len(_prompts(model, EMIT_MARK)) == 2 and result.facts["review_attempts_used"] == 1
    assert len(_prompts(model, QA_JUDGE_MARK)) == 4
    assert len(_prompts(model, OVERWATCH_JUDGE_MARK)) == 4
    assert "could not be read" not in (run / "reviews/follow-ups.md").read_text()
    assert (run / "reviews/overwatch-gate.txt").read_text().startswith("proceed: overwatch passed")


def test_a_verdict_still_unreadable_goes_on_with_a_note_the_pull_request_carries(tmp_path):
    """D3, R11: a judge whose verdict is unreadable again is not asked a third time. The
    QA gate goes on to the overwatch with a note; the overwatch gate goes on with a note
    that acceptance was not judged; and the pull request does not say the audit passed."""
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(_answers(_round(
        _tool("built"), judge=[Scripted("Looks fine to me."), Scripted("Still fine.")],
        overwatch=[Scripted("VERDICT: PASS (in prose)"), Scripted("VERDICT: PASS")])))
    run = tmp_path / "run"
    result = _design(run, model, repo)

    assert result.exit_ok, result.error
    assert len(_prompts(model, QA_JUDGE_MARK)) == 2
    assert len(_prompts(model, OVERWATCH_JUDGE_MARK)) == 2
    assert not result.facts.get("review_attempts_used")
    gate = (run / "reviews/qa-gate.txt").read_text()
    assert gate.startswith("proceed: the QA judge's verdict could not be read "
                           "(qa-judge.json holds no JSON object), also when it was asked again"), gate
    follow_ups = (run / "reviews/follow-ups.md").read_text()
    assert "the change went on to the overwatch without it. It wrote: Still fine." in follow_ups
    assert "acceptance was not judged, so check each ask and correction by hand" in follow_ups
    body = (run / delivery.BODY).read_text()
    assert "- **Reviews:** not all review steps completed." in body
    assert "overwatch audit passed" not in body


@pytest.mark.parametrize("allowed, verdicts", [(0, ["proceed"]), (1, ["reread", "proceed"]),
                                              (2, ["reread", "reread", "proceed"])])
def test_the_rereads_follow_their_knob(tmp_path, allowed, verdicts):
    """The reread count is code-side: a fact the gate keeps, capped by the frozen
    `judge_rereads` knob."""
    ctx = _gate_ctx(tmp_path, "not a verdict")
    ctx.knobs["judge_rereads"] = allowed
    assert [str(design_ops.gate_qa(ctx)) for _ in verdicts] == verdicts
    assert ctx.facts.get("qa_judge_rereads", 0) == allowed
    assert not ctx.facts.get("review_attempts_used")


# ── the other modes keep the parser gates ─────────────────────────────────────
@pytest.mark.parametrize("mode", [CreateMode, EditMode])
def test_create_and_edit_keep_their_gates(mode):
    """create and edit are unchanged: the parser gates, the overwatch writing the evidence
    map, its free audit round, and no judge (their lowering snapshots hold the rest)."""
    graph = mode().build()
    assert graph.nodes["gate_qa"].evaluator == "refactory_dsh.create_ops:gate_qa"
    assert graph.nodes["gate_overwatch"].evaluator == "refactory_dsh.create_ops:gate_overwatch"
    assert "qa_judge" not in graph.nodes and "audit_corrections" not in graph.nodes
    assert [(e.target, e.condition) for e in graph.out_edges("gate_overwatch")
            if e.condition == "audit"] == [("overwatch", "audit")]
    overwatch = graph.nodes["overwatch"]
    assert "Write intent-audit.json as" in overwatch.prompt_template
    assert evidence.AUDIT in overwatch.writes
    assert "correction-audit.json" in graph.nodes["gate_overwatch"].writes


def test_the_design_graph_has_judges_and_no_free_audit_round():
    graph = DesignMode().build()
    assert graph.nodes["synthesize_qa"].call == "refactory_dsh.design_ops:synthesize_qa"
    assert graph.nodes["gate_qa"].evaluator == "refactory_dsh.design_ops:gate_qa"
    assert graph.nodes["gate_overwatch"].evaluator == "refactory_dsh.design_ops:gate_overwatch"
    assert [e.target for e in graph.out_edges("synthesize_qa")] == ["qa_judge"]
    # A reread goes back to the judge alone; only a reloop reaches the builder, through
    # pick_corrections (dm-226).
    assert {(e.target, e.condition) for e in graph.out_edges("gate_qa")} == {
        ("pick_corrections", "reloop"), ("qa_judge", "reread"), ("audit_corrections", "proceed")}
    assert [e.target for e in graph.out_edges("audit_corrections")] == ["overwatch"]
    assert {(e.target, e.condition) for e in graph.out_edges("gate_overwatch")} == {
        ("pick_corrections", "reloop"), ("overwatch", "reread"), ("report", "proceed")}
    for gate, count in (("gate_qa", "qa_judge_rereads"), ("gate_overwatch", "overwatch_rereads")):
        assert count in graph.nodes[gate].produces and "judge_rereads" in graph.nodes[gate].knobs
    knob = DesignMode.declare()["judge_rereads"]
    assert (knob.default, knob.frozen) == (1, True)
    # The QA judge writes the evidence map the audit measures; the overwatch reads it.
    assert evidence.AUDIT in graph.nodes["qa_judge"].writes
    assert "Write intent-audit.json as" in graph.nodes["qa_judge"].prompt_template
    assert evidence.AUDIT not in graph.nodes["overwatch"].writes
    assert "Write intent-audit.json as" not in graph.nodes["overwatch"].prompt_template
    # The QA judge reads the whole design and the validation receipt.
    assert {"design.md", evidence.EVIDENCE} <= set(graph.nodes["qa_judge"].reads)
    # On the spine an agent's declared inputs must exist: no judge declares the optional
    # verifier's report, and the overwatch declares nothing a skipped QA judge or a skipped
    # correction audit would not leave.
    for node in ("qa_judge", "overwatch"):
        assert "reviews/verification.md" not in graph.nodes[node].reads
    for name in (judges.QA_JUDGE, evidence.AUDIT, "correction-audit.json"):
        assert name not in graph.nodes["overwatch"].reads


def test_the_judges_answer_in_their_verdicts_not_in_review_markers():
    """R17: neither judge inherits the review context that says to write `UNAVAILABLE:`
    into a report or to resolve errors in its own report; each is told that a missing
    dependency is a must-fix item in its JSON."""
    from refactory_dsh.authoring_evidence import FULL_REVIEW_CONTEXT

    assert "write `UNAVAILABLE:`" in FULL_REVIEW_CONTEXT
    graph = DesignMode().build()
    for node in ("qa_judge", "overwatch"):
        prompt = graph.nodes[node].prompt_template
        assert "write `UNAVAILABLE:`" not in prompt, node
        assert "in your own report" not in prompt, node
        assert "BLOCKING" not in prompt and "VERDICT:" not in prompt, node
        assert "must_fix item whose `what` starts with `UNAVAILABLE:`" in prompt, node
        assert '"severity": "critical" | "high" | "medium" | "low"' in prompt, node
    assert "{{runtime_evidence}}" in graph.nodes["overwatch"].prompt_template


# ── the judges' files ──────────────────────────────────────────────────────────
def test_a_judgment_is_one_json_object_and_a_redo_names_what_to_fix():
    """D9: the file's only JSON content is one object -- bare, fenced, or among prose.
    Two objects are unreadable; braces in prose are not objects."""
    verdict = qa_judgment("proceed", follow_ups=["docs"])
    for text in (verdict, "```json\n" + verdict + "\n```",
                 "Here is my verdict:\n```json\n" + verdict + "\n```\nThat is all {really}.",
                 "Verdict follows " + verdict + " -- done."):
        assert judges.parse(text, judges.QA_VERDICTS).follow_ups == ("docs",), text
    for text, why in (("VERDICT: PASS", "holds no JSON object"),
                      ("Example: " + qa_judgment("redo", must_fix=["x"]) + "\nMine: " + verdict,
                       "holds 2 JSON objects; it must hold exactly one"),
                      (json.dumps({"verdict": "maybe"}), "has verdict 'maybe'"),
                      (qa_judgment("redo"), "says redo but names nothing to fix"),
                      (json.dumps({"verdict": "proceed", "must_fix": "x"}), "must_fix is not a list")):
        with pytest.raises(judges.Unreadable, match=why):
            judges.parse(text, judges.QA_VERDICTS)
    parsed = judges.parse(overwatch_judgment("redo", corrections=[
        {"id": "ledger-3", "met": False, "evidence": "x"}, {"id": "ledger-4", "met": None}]),
        judges.OVERWATCH_VERDICTS)
    assert [item["id"] for item in parsed.unmet] == ["ledger-3"]
    assert [item["id"] for item in parsed.unjudged] == ["ledger-4"]
    # A must-fix item with no severity is high: a missing label never lets a defect through.
    [item] = judges.parse(qa_judgment("redo", must_fix=[{"what": "x"}]), judges.QA_VERDICTS).must_fix
    assert (item["severity"], item["breaks"], judges.blocks(item)) == ("high", "none", True)


def test_the_synthesis_carries_every_review_unread_and_the_verifier_only_when_it_reported(tmp_path):
    """R05: each review goes to the QA judge exactly as written, with nothing a parser read
    out of it, and the verifier only when it reported."""
    ctx = nn.RunContext(tmp_path, facts={"check": "python tool.py", "validation": "(the check passed)"})
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, f"{name}: BLOCKING: tool.py: quoted\nmore prose\n")
    ctx.write_text("design.md", "# T\n\n## Changes\n\n- tool.py\n\n## Docs\n\n- none\n")
    body = design_ops.synthesize_qa(ctx)["qa"]
    for name, who in design_ops.DESIGN_QA_REVIEWS:
        assert f"## {name} (the {who})\n\n{name}: BLOCKING: tool.py: quoted\nmore prose\n\n" in body
    assert "verification" not in body, "a verifier that left nothing is simply absent"
    assert "## The design's Changes\n\n- tool.py\n" in body
    ctx.write_text(design_ops.VERIFICATION, "the verifier's report\n")
    assert "the verifier's report" in design_ops.synthesize_qa(ctx)["qa"]


def test_each_round_starts_with_nothing_an_earlier_round_judged(tmp_path):
    """A skipped judge or audit leaves nothing for its gate or the overwatch: the last
    round's verdicts, evidence map and audit are set aside, kept, when the round starts."""
    ctx = nn.RunContext(tmp_path, facts={"check": "true"})
    for name in design_ops.JUDGING_FILES:
        ctx.write_text(name, f"{name} from the last round")
    produced = design_ops.synthesize_qa(ctx)
    assert not any((tmp_path / name).exists() for name in design_ops.JUDGING_FILES)
    [kept] = list((tmp_path / design_ops.JUDGING_HISTORY).iterdir())
    assert sorted(path.name for path in kept.iterdir()) == sorted(design_ops.JUDGING_FILES)
    assert (produced["qa_judge_rereads"], produced["overwatch_rereads"]) == (0, 0)
    assert produced["correction_audit"] == "(the correction audit has not run this round)"


def _gate_ctx(tmp_path: Path, judgment: str, **facts) -> nn.RunContext:
    """A run directory with this attempt's three reviews and the QA judge's file."""
    ctx = nn.RunContext(tmp_path, facts=facts)
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        ctx.write_text(name, "ran it; fine\n")
    ctx.write_text(judges.QA_JUDGE, judgment)
    return ctx


def _overwatch_ctx(tmp_path: Path, judgment: str, **facts) -> nn.RunContext:
    ctx = nn.RunContext(tmp_path, facts=facts)
    ctx.write_text(judges.OVERWATCH_VERDICT, judgment)
    return ctx


UNAVAILABLE_ITEM = {"what": "UNAVAILABLE: no browser to render the page; install chromium"}


@pytest.mark.parametrize("gate, context, text", [
    ("gate_qa", _gate_ctx, qa_judgment("redo", must_fix=[UNAVAILABLE_ITEM])),
    ("gate_overwatch", _overwatch_ctx, overwatch_judgment("redo", must_fix=[UNAVAILABLE_ITEM])),
    ("gate_qa", _gate_ctx, "UNAVAILABLE: no browser to render the page; install chromium"),
    ("gate_overwatch", _overwatch_ctx,
     "I could not judge.\nUNAVAILABLE: no browser to render the page; install chromium\n"),
])
def test_a_judge_that_needs_an_unavailable_tool_stops_the_run(tmp_path, gate, context, text):
    """R17: a judge says a dependency is missing in a must-fix item, or its file holds no
    verdict but an `UNAVAILABLE:` line, as create's overwatch report would, and the run
    stops for a person instead of sending the builder a defect it cannot fix, or reading
    the judge again."""
    ctx = context(tmp_path, text)
    with pytest.raises(ValueError, match="validation unavailable: UNAVAILABLE: no browser"):
        getattr(design_ops, gate)(ctx)
    assert not ctx.facts.get("review_attempts_used") and not ctx.facts.get("qa_judge_rereads")


@pytest.mark.parametrize("gate, context, verdict", [
    ("gate_qa", _gate_ctx, qa_judgment("proceed")),
    ("gate_overwatch", _overwatch_ctx, overwatch_judgment("pass")),
])
def test_an_unavailable_line_beside_a_readable_verdict_decides_nothing(tmp_path, gate, context, verdict):
    """A readable verdict decides: prose beside it that starts `UNAVAILABLE:` is not a stop.
    A judge that needs a tool says so in a must-fix item, which its gate stops on."""
    ctx = context(tmp_path, "UNAVAILABLE: none, all tools present\n" + verdict)
    assert getattr(design_ops, gate)(ctx) == "proceed"


def test_a_review_that_says_a_tool_is_unavailable_stops_the_run(tmp_path):
    """R14: as before, a review line starting `UNAVAILABLE:` stops the run."""
    ctx = _gate_ctx(tmp_path, qa_judgment("proceed"))
    ctx.write_text("reviews/qa.md", "ran what I could\nUNAVAILABLE: no display; run under xvfb\n")
    with pytest.raises(ValueError, match="validation unavailable: UNAVAILABLE: no display"):
        design_ops.gate_qa(ctx)


def test_a_review_that_left_nothing_sends_the_change_back_whatever_the_judge_says(tmp_path):
    """An absent review is not a passed one: that stays structural, as it was."""
    ctx = _gate_ctx(tmp_path, qa_judgment("proceed"))
    ctx.write_text("reviews/qa.md", "")
    verdict = design_ops.gate_qa(ctx)
    assert verdict == "reloop" and "a review did not run: reviews/qa.md" in verdict.reason
    assert ctx.facts["review_attempts_used"] == 1


@pytest.mark.parametrize("gate, context, text", [
    ("gate_qa", _gate_ctx, qa_judgment("proceed")),
    ("gate_overwatch", _overwatch_ctx, overwatch_judgment("pass")),
])
def test_a_stale_sign_off_sends_the_change_back_whatever_the_judge_says(tmp_path, gate, context, text):
    """R13, R18: a review or overwatch whose sign-off is not current for this attempt --
    here, runtime evidence the run recorded but cannot find -- sends the change back even
    when its judge proceeds or passes."""
    ctx = context(tmp_path, text, runtime_evidence="recorded")
    verdict = getattr(design_ops, gate)(ctx)
    assert verdict == "reloop" and "authoritative runtime evidence is missing" in verdict.reason
    assert ctx.facts["review_attempts_used"] == 1


MINOR = {"source": "code_reviewer", "where": "tool.py:3", "what": "no --quiet flag",
         "why": "nice to have", "severity": "low", "breaks": "none"}


@pytest.mark.parametrize("item, back", [
    (MINOR, False),
    ({**MINOR, "severity": "medium"}, False),
    ({**MINOR, "severity": "high"}, True),
    ({**MINOR, "severity": "critical"}, True),
    ({**MINOR, "severity": "medium", "breaks": "ask"}, True),
    ({**MINOR, "severity": "low", "breaks": "correction"}, True),
    ({**MINOR, "severity": "low", "breaks": "health"}, True),
    ({**MINOR, "severity": "low", "breaks": "regression"}, True),
    ({key: value for key, value in MINOR.items() if key != "severity"}, True),
])
def test_only_what_holds_the_change_back_sends_it_back(tmp_path, item, back):
    """The severity bar (remote-factory design-v2's rule): a redo goes back only for an
    item that is critical or high, or breaks an ask, a correction or the health check.
    Otherwise the change goes on and the items are follow-ups (R15)."""
    ctx = _gate_ctx(tmp_path, qa_judgment("redo", must_fix=[item]))
    verdict = design_ops.gate_qa(ctx)
    assert str(verdict) == ("reloop" if back else "proceed")
    if not back:
        follow_ups = (tmp_path / "reviews/follow-ups.md").read_text()
        assert "[code_reviewer] tool.py:3: no --quiet flag" in follow_ups
        assert "asked for a redo, but nothing it named is critical or high" in follow_ups


def test_a_proceed_lists_the_must_fix_items_it_left(tmp_path):
    """R15: a proceed that still names must-fix items lists them on the pull request."""
    ctx = _gate_ctx(tmp_path, qa_judgment("proceed", must_fix=[{**MINOR, "severity": "high"}]))
    assert design_ops.gate_qa(ctx) == "proceed"
    assert ("[code_reviewer] tool.py:3: no --quiet flag (why: nice to have) (high) "
            "(the QA judge proceeded with this listed as must-fix)"
            in (tmp_path / "reviews/follow-ups.md").read_text())


def test_an_overwatch_pass_with_an_unmet_correction_goes_back_naming_it(tmp_path):
    """D4: a correction the overwatch finds not met does not ship, whatever its verdict:
    the change goes back with the correction and the overwatch's evidence, and spends a
    review attempt."""
    ctx = _overwatch_ctx(tmp_path, overwatch_judgment("pass", corrections=[
        {"id": "ledger-2", "met": False, "evidence": "calc.py:2 still returns x > 0"}]))
    verdict = design_ops.gate_overwatch(ctx)
    assert verdict == "reloop"
    assert "- correction ledger-2: NOT met — calc.py:2 still returns x > 0" in verdict.reason
    assert ctx.facts["review_attempts_used"] == 1
    assert "ledger-2: NOT met" in ctx.facts["overwatch_review"]


@pytest.mark.parametrize("corrections, shown", [
    ([{"id": "ledger-2", "met": False, "evidence": "UNMET-EVIDENCE-42"}],
     "- correction ledger-2: NOT met — UNMET-EVIDENCE-42"),
    ([{"id": "ledger-3", "met": None}], "- correction ledger-3: not judged"),
])
def test_an_overwatch_redo_for_a_correction_tells_the_builder_which_and_why(tmp_path, corrections, shown):
    """R09, R20: a redo that names only a correction, unmet or not judged, sends the
    builder that correction with the overwatch's evidence."""
    ctx = _overwatch_ctx(tmp_path, overwatch_judgment("redo", corrections=corrections))
    verdict = design_ops.gate_overwatch(ctx)
    assert verdict == "reloop" and shown in verdict.reason
    assert shown in ctx.facts["validation"]


def test_a_minor_overwatch_redo_goes_on_but_an_unmet_correction_still_goes_back(tmp_path):
    """The severity bar applies to the overwatch's must-fix items, except that an unmet
    correction always sends the change back."""
    ctx = _overwatch_ctx(tmp_path, overwatch_judgment("redo", must_fix=[MINOR]))
    assert design_ops.gate_overwatch(ctx) == "proceed"
    assert "no --quiet flag" in (tmp_path / "reviews/follow-ups.md").read_text()
    ctx = _overwatch_ctx(tmp_path, overwatch_judgment("redo", must_fix=[MINOR], corrections=[
        {"id": "ledger-2", "met": False, "evidence": "e"}]))
    assert design_ops.gate_overwatch(ctx) == "reloop"


def test_a_pass_lists_what_it_did_not_judge_and_the_must_fix_it_left(tmp_path, monkeypatch):
    """R10, R16: a pass ships, and the pull request names every correction the overwatch
    did not judge -- listed without a decision, or not listed at all -- and every must-fix
    item it left."""
    monkeypatch.setattr(evidence, "current_problems", lambda ctx, workers: [])
    ctx = _overwatch_ctx(tmp_path, overwatch_judgment(
        "pass", must_fix=[{**MINOR, "severity": "high"}],
        corrections=[{"id": "ledger-2", "met": None, "evidence": "could not run it"}]))
    ctx.write_json(evidence.EVIDENCE, {"requirements": [
        {"id": "ledger-2", "kind": "correction", "text": "a"},
        {"id": "ledger-3", "kind": "constraint", "text": "b"}]})
    assert design_ops.gate_overwatch(ctx) == "proceed"
    follow_ups = (tmp_path / "reviews/follow-ups.md").read_text()
    assert ("correction ledger-2: not judged — could not run it (the overwatch could not judge it)"
            in follow_ups)
    assert "correction ledger-3: the overwatch did not judge it" in follow_ups
    assert "no --quiet flag" in follow_ups and "(the overwatch passed with this listed as must-fix)" in follow_ups


def test_a_verdict_among_prose_is_read(tmp_path):
    """D9: the overwatch's one JSON object, fenced among prose, decides the gate."""
    ctx = _overwatch_ctx(tmp_path, "Here is my verdict:\n```json\n" + overwatch_judgment("pass")
                         + "\n```\nAll asks have evidence.")
    assert design_ops.gate_overwatch(ctx) == "proceed"
    assert (tmp_path / "reviews/overwatch-gate.txt").read_text().startswith("proceed: overwatch passed")


def test_an_unreadable_overwatch_verdict_is_read_again_then_goes_on_with_a_note(tmp_path):
    """Sending the builder back cannot repair a judge's file: it is read again once, and
    then the change goes on, with the pull request saying acceptance was not judged and
    what the overwatch wrote."""
    ctx = _overwatch_ctx(tmp_path, "VERDICT: FAIL\nthe ask has no evidence")
    first = design_ops.gate_overwatch(ctx)
    assert first == "reread" and "holds no JSON object" in first.reason
    assert ctx.facts["overwatch_rereads"] == 1
    assert "overwatch-verdict.json holds no JSON object" in ctx.facts["overwatch_reread"]
    assert design_ops.gate_overwatch(ctx) == "proceed"
    notes = (tmp_path / "reviews/follow-ups.md").read_text()
    assert "acceptance was not judged" in notes
    assert "It wrote: VERDICT: FAIL the ask has no evidence" in notes
    assert not ctx.facts.get("review_attempts_used")


def test_the_audit_writes_its_file_every_round_even_with_nothing_to_measure(tmp_path):
    """A round with nothing to audit still writes this round's audit, for the overwatch."""
    ctx = nn.RunContext(tmp_path)
    ctx.write_json("correction-audit.json", [{"id": "ledger-9", "clauses": []}])
    produced = design_ops.audit_corrections(ctx)
    assert json.loads((tmp_path / "correction-audit.json").read_text()) == []
    assert produced["correction_audit"].startswith("No correction or constraint")


def test_the_report_refuses_a_sign_off_that_is_no_longer_current(tmp_path):
    """R12: the judged report still refuses when a review's sign-off is not current: here
    the approved design changed after the reviews signed off."""
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(_answers(_round(_tool("built"), overwatch=overwatch_judgment("pass"))))
    run = tmp_path / "run"
    result = _design(run, model, repo)
    assert result.exit_ok, result.error
    # The knobs the check recorded its settings under, as the run's own context has them.
    settings = json.loads((run / evidence.EVIDENCE).read_text())["settings"]
    ctx = nn.RunContext(run, knobs=settings, facts=result.facts, enforce_writes=False)
    design_ops.report_judged(ctx)
    (run / "design.md").write_text((run / "design.md").read_text() + "\nchanged after review\n")
    with pytest.raises(ValueError, match="authoring sign-off is incomplete or stale"):
        design_ops.report_judged(ctx)


# ── the gates on the spine: each its own process, through refactory_dsh.run ───
def _spine_run(tmp_path: Path, judgment: str, **knobs: int) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    mode = DesignMode()
    state = mode.state_dict()
    for key, value in knobs.items():
        state["knobs"][key]["value"] = value
    mode.load_state_dict(state)
    lowering.write_declarations(mode.state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": NAME, "brief": "a tool", "check": "true"}))
    for name, _ in design_ops.DESIGN_QA_REVIEWS:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("BLOCKING: tool.py: quoted in prose\n")
    (run / judges.QA_JUDGE).write_text(judgment)
    return run


def _on_the_spine(run: Path, monkeypatch, node: str, kind: str,
                  knobs: dict[str, int]) -> tuple[int, str, str]:
    """Run one design_ops node as the spine does: its own command, its declaration in
    DSH_WORKFLOW_NODE, its facts read from and written back to facts.json."""
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node, "type": kind}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps(knobs))
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    code = runner_main([f"{design_ops.__name__}:{node}", "--node", node])
    monkeypatch.undo()
    return code, out.getvalue(), err.getvalue()


def _gate_qa_on_the_spine(run: Path, monkeypatch, knobs: dict[str, int]) -> tuple[int, str, str]:
    return _on_the_spine(run, monkeypatch, "gate_qa", "gate", knobs)


def test_gate_qa_on_the_spine_routes_on_the_judge_file(tmp_path, monkeypatch):
    """The spine runs gate_qa as a process reading qa-judge.json: the verdict is the last
    line it prints, the must-fix item the reason before it, and the facts it set are
    written back for the next node."""
    knobs = {"review_attempts": 3, "check_attempts": 3}
    must_fix = {"source": "qa_director", "where": "tool.py:2", "what": "crashes on empty input",
                "why": "the brief asks for any input"}
    run = _spine_run(tmp_path, qa_judgment("redo", must_fix=[must_fix]))
    code, out, err = _gate_qa_on_the_spine(run, monkeypatch, knobs)
    assert code == 0, err
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[-1] == "reloop"
    assert "tool.py:2: crashes on empty input" in out
    facts = json.loads((run / "facts.json").read_text())
    assert facts["review_attempts_used"] == 1
    assert "crashes on empty input" in facts["qa_gate"]

    (run / judges.QA_JUDGE).write_text(qa_judgment("proceed", follow_ups=["a docs line"]))
    code, out, err = _gate_qa_on_the_spine(run, monkeypatch, knobs)
    assert code == 0, err
    assert out.split()[-1] == "proceed"
    assert "a docs line (qa)" in (run / "reviews/follow-ups.md").read_text()


def test_gate_qa_on_the_spine_stops_at_the_review_limit(tmp_path, monkeypatch):
    run = _spine_run(tmp_path, qa_judgment("redo", must_fix=[{"what": "still broken"}]))
    code, _, err = _gate_qa_on_the_spine(run, monkeypatch, {"review_attempts": 1, "check_attempts": 3})
    assert code == 3, err
    assert FAILURE_MARKER in err and "review reloop limit reached after 1" in err


def test_gate_qa_on_the_spine_proceeds_without_a_judge_file(tmp_path, monkeypatch):
    """A judge that left no file does not stall the gate: it is asked again once, with no
    builder round, and then the gate proceeds with the note."""
    run = _spine_run(tmp_path, "")
    (run / judges.QA_JUDGE).unlink()
    knobs = {"review_attempts": 3, "judge_rereads": 1}
    code, out, err = _gate_qa_on_the_spine(run, monkeypatch, knobs)
    assert code == 0, err
    assert out.split()[-1] == "reread" and "qa-judge.json was not written" in out
    facts = json.loads((run / "facts.json").read_text())
    assert facts["qa_judge_rereads"] == 1 and not facts.get("review_attempts_used")
    assert "qa-judge.json was not written" in facts["qa_judge_reread"]
    code, out, err = _gate_qa_on_the_spine(run, monkeypatch, knobs)
    assert code == 0, err
    assert out.split()[-1] == "proceed"
    assert "qa-judge.json was not written" in (run / "reviews/qa-gate.txt").read_text()


def test_the_qa_tail_runs_node_by_node_on_the_spine(tmp_path, monkeypatch):
    """synthesize_qa, gate_qa, audit_corrections and gate_overwatch, each its own process
    with the lowered graph's declarations: every fact and file they set is declared, and
    gate_overwatch routes on overwatch-verdict.json."""
    knobs = {"review_attempts": 3, "check_attempts": 3}
    run = _spine_run(tmp_path, qa_judgment("proceed"))
    (run / "design.md").write_text("# T\n\n## Changes\n\n- tool.py\n")
    (run / "check-record.json").write_text(json.dumps({"status": "pass"}))
    code, out, err = _on_the_spine(run, monkeypatch, "synthesize_qa", "op", knobs)
    assert code == 0, err
    # The round starts with nothing an earlier round judged: _spine_run's judge file is
    # set aside, and the judge writes this round's.
    assert not (run / judges.QA_JUDGE).exists()
    assert json.loads((run / "facts.json").read_text())["correction_audit"] == (
        "(the correction audit has not run this round)")
    (run / judges.QA_JUDGE).write_text(qa_judgment("proceed"))
    for node, kind in (("gate_qa", "gate"), ("audit_corrections", "op")):
        code, out, err = _on_the_spine(run, monkeypatch, node, kind, knobs)
        assert code == 0, (node, err)
    assert "## The design's Changes\n\n- tool.py" in (run / "reviews/qa-synthesized.md").read_text()
    assert json.loads((run / "correction-audit.json").read_text()) == []
    facts = json.loads((run / "facts.json").read_text())
    assert facts["qa_missing"] == 0 and facts["correction_audit"].startswith("No correction")

    (run / judges.OVERWATCH_VERDICT).write_text(overwatch_judgment("redo", must_fix=[
        {"source": "overwatch", "where": "tool.py:1", "what": "no evidence for the ask"}]))
    code, out, err = _on_the_spine(run, monkeypatch, "gate_overwatch", "gate", knobs)
    assert code == 0, err
    assert out.split()[-1] == "reloop" and "no evidence for the ask" in out
    assert json.loads((run / "facts.json").read_text())["review_attempts_used"] == 1

    (run / judges.OVERWATCH_VERDICT).write_text(overwatch_judgment("pass"))
    code, out, err = _on_the_spine(run, monkeypatch, "gate_overwatch", "gate", knobs)
    assert code == 0, err
    assert out.split()[-1] == "proceed"
    assert (run / "reviews/overwatch-gate.txt").read_text().startswith("proceed: overwatch passed")


def test_a_skipped_judge_never_reads_an_earlier_rounds_verdict_on_the_spine(tmp_path, monkeypatch):
    """A person skips the QA judge on round 2: the gate does not read round 1's redo
    again. The round's start set it aside, so the judge is asked again instead, with no
    builder round."""
    knobs = {"review_attempts": 3, "check_attempts": 3}
    run = _spine_run(tmp_path, qa_judgment("redo", must_fix=[{"what": "round one's defect"}]))
    code, out, _ = _gate_qa_on_the_spine(run, monkeypatch, knobs)
    assert out.split()[-1] == "reloop"
    code, out, err = _on_the_spine(run, monkeypatch, "synthesize_qa", "op", knobs)
    assert code == 0, err
    code, out, err = _gate_qa_on_the_spine(run, monkeypatch, knobs)
    assert code == 0, err
    assert out.split()[-1] == "reread" and "round one's defect" not in out
    assert json.loads((run / "facts.json").read_text())["review_attempts_used"] == 1


def test_an_unmet_correction_reaches_the_builders_prompt_on_the_spine(tmp_path, monkeypatch):
    """R09 on the spine: gate_overwatch's reason and facts carry an unmet correction and the
    overwatch's evidence into emit's rendered prompt, across processes."""
    from refactory_dsh.nn.executor import render

    run = _spine_run(tmp_path, qa_judgment("proceed"))
    (run / judges.OVERWATCH_VERDICT).write_text(overwatch_judgment("redo", corrections=[
        {"id": "ledger-2", "met": False, "evidence": "UNMET-EVIDENCE-42"}]))
    code, out, err = _on_the_spine(run, monkeypatch, "gate_overwatch", "gate",
                                   {"review_attempts": 3, "check_attempts": 3})
    assert code == 0, err
    assert out.split()[-1] == "reloop"
    facts = json.loads((run / "facts.json").read_text())
    emit = DesignMode().build().nodes["emit"].prompt_template
    assert "ledger-2: NOT met — UNMET-EVIDENCE-42" in render(emit, nn.RunContext(run, facts=facts))


@pytest.mark.parametrize("mode, node, answer_file, other", [
    (DesignMode, "qa_judge", judges.QA_JUDGE, evidence.AUDIT),
    (DesignMode, "overwatch", judges.OVERWATCH_VERDICT, None),
    # create's overwatch too: its report, not the evidence map it also writes.
    (CreateMode, "overwatch", "reviews/overwatch.md", evidence.AUDIT),
])
def test_a_judge_that_answers_in_its_message_is_captured_into_its_verdict_file(
        tmp_path, monkeypatch, mode, node, answer_file, other):
    """Spine and walker parity: the spine's worker captures an answer the agent did not
    write into the node's first declared output, as the walker does -- the verdict file,
    not whichever declared file sorts first (intent-audit.json for the QA judge, and for
    create's overwatch, whose report is reviews/overwatch.md)."""
    import shlex

    from refactory_dsh import workers
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.worker_node import execute

    verdict = (qa_judgment("proceed") if node == "qa_judge" else overwatch_judgment("pass")
               if mode is DesignMode else "Every ask has evidence.\n\nVERDICT: PASS")

    def complete(self, prompt, *, system="", workdir=None):
        if other:
            (Path(workdir) / other).write_text(json.dumps({"requirements": []}))
        return verdict

    monkeypatch.setattr(DshModelClient, "complete", complete)
    monkeypatch.setenv("REFACTORY_WORKER_BACKEND", "dsh")
    run = tmp_path / "run"
    run.mkdir()
    state = mode().state_dict()
    lowering.write_declarations(state, run)
    lowered = workers.lower_workflow(lowering.to_workflow(state), run,
                                     DshModelClient(python_bin=Path(sys.executable).parent).context(),
                                     lowering.node_declarations(state))
    spec = json.loads(Path(shlex.split(lowered.nodes[node].dispatch_command)[-1]).read_text())
    assert spec["writes"][0] == answer_file
    for name in spec["reads"]:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("{}")
    (run / evidence.EVIDENCE).write_text(json.dumps({"identity": {"root": str(run)}}))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node, "type": "AgentNode"}))
    receipt = execute(spec, run)
    assert receipt["status"] == "delivered", receipt
    assert receipt["captured_final"] == answer_file
    assert (run / answer_file).read_text() == verdict
