"""A design run works in its own worktree of its target, end to end (#351).

One test per path a person takes: a new idea with no repository, a change to a
repository they have checked out, the same idea again, a base other than HEAD, an
attempt that fails its check and is redone, and a run with its worktree turned off.
Then the edits an attempt makes (#330 I4): a small change to a long file, a retry that
edits what the last attempt wrote, and an emission materialise cannot apply, which goes
back to emit rather than ending the run. Last, design_fast, which takes the same steps
(`repo_change`) and delivers nothing. A fake model answers; the git repositories, the
worktrees, the check and the registry are real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pytest

from refactory_dsh import (cli, console, design_ops, interactive, lowering, nn, projects, registry, repo_change,
                           worktree)
from refactory_dsh import run as node_runner
from refactory_dsh.create import CreateMode as CreateModeForGraph
from refactory_dsh.design import DesignMode
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.nn import Verdict
from refactory_dsh.paths import REPO_ROOT
from refactory_dsh.projects import projects_dir
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
                           RESEARCH_ANSWER, VERIFICATION_ANSWER,
                           REVIEW_ANSWERS, FakeModel,
                           StaleEmitModel, git, git_repo, render, stub)

NAME = "worktree_tool"


# ── graph edges for audit reloop and refused emission ──────────────────────


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def test_gate_overwatch_has_no_audit_edge_in_design_graph():
    """Design mode's gate_overwatch routes on the overwatch's verdict only: the evidence
    map's problems are the overwatch's evidence (design_ops.audit_corrections), not a
    free round back to it."""
    graph = DesignMode().build()
    edges = graph.out_edges("gate_overwatch")
    assert [e for e in edges if e.condition == "audit"] == []


def test_gate_overwatch_audit_edge_exists_in_create_graph():
    """gate_overwatch has an 'audit' edge back to overwatch in the create graph."""
    graph = CreateModeForGraph().build()
    edges = graph.out_edges("gate_overwatch")
    audit_edges = [e for e in edges if e.condition == "audit"]
    assert len(audit_edges) == 1
    assert audit_edges[0].target == "overwatch"


def test_refused_emission_does_not_spend_attempt_design(tmp_path, monkeypatch):
    """check_artifact returns Verdict('reloop') without calling repair() for a refused emission."""
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "edit text not found")
    ctx = nn.RunContext(tmp_path, facts={"check": "python tool.py"},
                        knobs={"attempts": 3})
    ctx._declared = ("check.txt",)
    ctx._declared_facts = ("validation", "attempts_used")
    ctx.enforce_writes = False
    result = design_ops.check_artifact(ctx)
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert "materialise could not apply" in result.reason
    assert not ctx.facts.get("attempts_used"), "refused emission must not spend an attempt"


@pytest.fixture
def repo(tmp_path) -> Path:
    return git_repo(tmp_path / "repo")


def _answers(*emits: str) -> list[str]:
    """Every answer a design run needs, in the order its graph asks.

    pick_corrections is a gate (no model answer needed).  When there are no
    corrections, pick_corrections returns "skip" and apply_corrections does
    not run.
    """
    return [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER, *emits,
            *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]


def _mode(**knobs) -> DesignMode:
    """Design mode with some of its knobs set, as a person sets them for one run."""
    mode = DesignMode()
    state = mode.state_dict()
    for name, value in knobs.items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    return mode


def _design(workdir: Path, model: FakeModel, mode: DesignMode | None = None, **task: str):
    return (mode or DesignMode()).run(
        workdir, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "a tool", "check": "python tool.py", **task},
    )


def _report(workdir: Path) -> dict:
    return json.loads((workdir / "design.json").read_text())


def _edit(name: str, *pairs: tuple[str, str]) -> str:
    """An ``===EDIT:`` block: each pair is the lines to find and the lines put in their place."""
    body = "".join(f"{repo_change.SEARCH}\n{search}{repo_change.DIVIDER}\n{replacement}{repo_change.REPLACE}\n"
                   for search, replacement in pairs)
    return f"===EDIT: {name}===\n{body}"


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


# ── the paths a person takes ───────────────────────────────────────────────
def test_a_run_with_no_target_starts_a_new_project_and_commits_its_work_on_a_branch(tmp_path):
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": "print('built')\n"}))))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    project = (projects_dir() / NAME).resolve()
    assert payload["target"] == str(project)
    assert payload["base"] == git(project, "rev-parse", "main")
    assert git(project, "rev-list", "--count", "main") == "1", "main keeps only its empty first commit"
    assert git(project, "rev-list", "--count", f"{payload['base']}..{payload['branch']}") == "1"
    assert git(project, "ls-tree", "--name-only", payload["branch"]) == "tool.py"
    assert "tool.py" in payload["diff_stat"]
    [entry] = projects.load()["projects"]
    assert (entry["kind"], entry["runs"][-1]["branch"], entry["runs"][-1]["status"]) == (
        "new", payload["branch"], "succeeded")


def test_the_same_idea_again_is_a_new_branch_of_the_same_project_from_main(tmp_path):
    first = _design(tmp_path / "one", FakeModel(_answers(render({"tool.py": "print(1)\n"}))))
    second = _design(tmp_path / "two", FakeModel(_answers(render({"tool.py": "print(2)\n"}))))
    assert first.exit_ok and second.exit_ok, (first.error, second.error)
    one, two = _report(tmp_path / "one"), _report(tmp_path / "two")
    assert one["target"] == two["target"] and one["branch"] != two["branch"]
    assert one["base"] == two["base"], "the second run starts from main, not from the first run's branch"
    [entry] = projects.load()["projects"]
    assert [note["status"] for note in entry["runs"]] == ["succeeded", "succeeded"]


def test_a_run_on_a_repository_leaves_the_persons_checkout_and_uncommitted_work_alone(repo, tmp_path, capsys):
    """Its work is a branch of their repository; their checkout, edits included, is untouched (#330 I1)."""
    (repo / "app.py").write_text("print('mine, not committed')\n")
    (repo / "notes.txt").write_text("untracked\n")
    head = git(repo, "rev-parse", "HEAD")
    # This check passes only where app.py is as committed: the worktree, from its pinned base.
    tool = "import pathlib\nassert pathlib.Path('app.py').read_text() == \"print('app')\\n\"\n"
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": tool}))), target=str(repo))
    assert result.exit_ok, result.error
    assert (git(repo, "branch", "--show-current"), git(repo, "rev-parse", "HEAD")) == ("main", head)
    assert (repo / "app.py").read_text() == "print('mine, not committed')\n"
    assert (repo / "notes.txt").exists() and not (repo / "tool.py").exists()
    payload = _report(tmp_path / "run")
    assert (payload["target"], payload["base"]) == (str(repo.resolve()), head)
    assert git(repo, "ls-tree", "--name-only", payload["branch"]).splitlines() == ["app.py", "tool.py"]
    assert "uncommitted change(s) in" in capsys.readouterr().err
    [entry] = projects.load()["projects"]
    assert entry["kind"] == "existing"


def test_a_run_from_another_base_works_from_that_commit(repo, tmp_path):
    first = git(repo, "rev-parse", "HEAD")
    (repo / "later.py").write_text("x = 1\n")
    git(repo, "add", "later.py")
    git(repo, "commit", "-q", "-m", "later")
    tool = "import pathlib\nassert not pathlib.Path('later.py').exists()\n"
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": tool}))), target=str(repo),
                     base=first)
    assert result.exit_ok, result.error
    assert _report(tmp_path / "run")["base"] == first


def test_a_base_that_names_nothing_is_refused_before_anything_is_pruned_or_made(repo, tmp_path, monkeypatch):
    ended = tmp_path / "ended"
    ended.mkdir()
    earlier = worktree.create(repo, "HEAD", "0123abcd0123", ended)
    (ended / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "failed"}))
    monkeypatch.setenv("REFACTORY_WORKTREE_IDLE_HOURS", "0")
    result = _design(tmp_path / "run", FakeModel(_answers()), target=str(repo), base="no-such-branch")
    assert result.exit_ok is False
    assert "the base 'no-such-branch' is not a commit" in (result.error or "")
    assert Path(earlier.path).exists() and worktree.recorded(tmp_path / "run") is None


def test_a_redo_of_prepare_keeps_the_base_it_pinned_even_once_that_branch_is_gone(repo, tmp_path):
    git(repo, "branch", "start")
    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory" / "invocation.json").write_text(json.dumps({"id": "0123456789abcdef"}))
    task = {"name": NAME, "brief": "a tool", "check": "python tool.py", "target": str(repo), "base": "start"}
    first = design_ops.prepare_design(nn.RunContext(run, model=FakeModel([]), task=task))
    git(repo, "branch", "-D", "start")
    again = design_ops.prepare_design(nn.RunContext(run, model=FakeModel([]), task=task))
    assert (again["worktree"], again["base"]) == (first["worktree"], first["base"])


def test_a_target_inside_a_repository_is_refused_before_anything_is_made(repo, tmp_path):
    (repo / "sub").mkdir()
    result = _design(tmp_path / "run", FakeModel(_answers()), target=str(repo / "sub"))
    assert result.exit_ok is False
    assert f"pass its root, {repo.resolve()}" in (result.error or "")
    assert not worktree.slot(repo).exists() and projects.load()["projects"] == []


def test_facts_json_records_final_attempts_used_on_failure(repo, tmp_path, monkeypatch):
    """run_node's except clause writes facts.json so attempts_used on disk agrees with the error.

    Without it (after removing _persist_facts from create_ops), no writer would persist
    the facts a raising node set, and the on-disk attempts_used would stay at 0.
    """
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef999999", run)
    lowering.write_declarations(DesignMode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": NAME, "brief": "a tool", "check": "exit 1"}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 1}))
    (run / "emit.txt").write_text(render({"tool.py": "print('will fail check')\n"}))
    node_runner.run_node(f"{repo_change.OPS}:materialise", workdir=run, node_id="materialise")
    # Before the check, attempts_used is absent from facts.json on disk.
    assert "attempts_used" not in json.loads((run / "facts.json").read_text())
    with pytest.raises(ValueError, match=r"check attempts exhausted after 1 check failure\(s\)"):
        node_runner.run_node(f"{design_ops.__name__}:check_artifact", workdir=run, node_id="check")
    # run_node's except clause wrote facts.json; the on-disk value agrees with the message.
    saved = json.loads((run / "facts.json").read_text())
    assert saved["attempts_used"] == 1


def test_an_attempt_that_fails_its_check_is_redone_as_a_second_commit_on_top_of_it(tmp_path):
    """The retry changes only what it needs to; a file of the first attempt goes when it is deleted."""
    failing = render({"tool.py": "raise SystemExit(1)\n", "draft.py": "x = 1\n", "keep.py": "y = 2\n"})
    fixed = render({"tool.py": "print('fixed')\n"}) + "===DELETE: draft.py===\n"
    result = _design(tmp_path / "run", FakeModel(_answers(failing, fixed)))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    project = Path(payload["target"])
    assert git(project, "log", "--format=%s", f"{payload['base']}..{payload['branch']}").splitlines() == [
        f"{NAME}: attempt 2", f"{NAME}: attempt 1"]
    assert git(project, "ls-tree", "--name-only", payload["branch"]).split() == ["keep.py", "tool.py"]
    assert "draft.py" not in (tmp_path / "run" / "changes.diff").read_text()
    assert payload["files"] == ["keep.py", "tool.py"], "the report names what the branch delivers, not one attempt"


def test_a_check_that_rewrites_a_tracked_file_does_not_wedge_the_run(repo, tmp_path, capsys):
    """What the check changes stays in its check copy: the run's worktree, where the
    reviewers are, never sees it, and the next attempt's check starts from its commit."""
    tool = ("import os, pathlib\n"
            "pathlib.Path('app.py').write_text('rewritten by the check\\n')\n"
            "marker = pathlib.Path(os.environ['HOME']) / 'second-attempt'\n"
            "if not marker.exists():\n"
            "    marker.write_text('')\n"
            "    raise SystemExit('the first attempt is refused on purpose')\n")
    emit1 = render({"tool.py": tool})
    emit2 = render({"tool.py": tool, "note.txt": "retry\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emit1, emit2)), target=str(repo))
    assert result.exit_ok, result.error
    assert git(repo, "show", f"{_report(tmp_path / 'run')['branch']}:app.py") == "print('app')"
    warned = capsys.readouterr().err
    assert "the check changed tracked files in the run's worktree" not in warned
    assert "are not part of its branch" not in warned, "the run's worktree holds only its commits"


# ── edits, and an emission materialise cannot apply (#330 I4, E5) ──────────
def test_an_edit_changes_only_the_lines_it_names_in_a_long_file(repo, tmp_path):
    """A one-line change to a long file is a one-line diff: nothing else is retyped."""
    _commit(repo, "long.py", "".join(f"value_{index} = {index}\n" for index in range(300)))
    emission = (render({"tool.py": "import long\nassert long.value_150 == 'changed'\n"})
                + _edit("long.py", ("value_150 = 150\n", "value_150 = 'changed'\n")))
    result = _design(tmp_path / "run", FakeModel(_answers(emission)), target=str(repo))
    assert result.exit_ok, result.error
    branch = _report(tmp_path / "run")["branch"]
    assert git(repo, "diff", "--numstat", f"main..{branch}", "--", "long.py").split()[:2] == ["1", "1"]


def test_a_retry_edits_the_file_its_last_attempt_wrote(tmp_path):
    failing = render({"tool.py": "x = 1\nraise SystemExit('not yet')\n"})
    fixed = _edit("tool.py", ("raise SystemExit('not yet')\n", "print('fixed', x)\n"))
    result = _design(tmp_path / "run", FakeModel(_answers(failing, fixed)))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert git(Path(payload["target"]), "show", f"{payload['branch']}:tool.py") == "x = 1\nprint('fixed', x)"


class _WorkspaceBuilder(FakeModel):
    """A builder at work in its workspace: ``first`` and ``retry`` are what it does there
    on its first and second dispatch, beside its scripted emission's changes."""

    def __init__(self, answers: list[str], first: Callable[[Path], object],
                 retry: Callable[[Path], object] = lambda _: None) -> None:
        super().__init__(answers)
        self.steps = [first, retry]

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env, confine=confine)
        if cwd is not None and "Write the artifact `" in prompt and self.steps:
            self.steps.pop(0)(Path(str(cwd)))
        return answer


def _binary(clone: Path) -> None:
    (clone / "logo.png").write_bytes(b"\x89PNG\0\0")


def _symlink(clone: Path) -> None:
    (clone / "link.py").symlink_to("app.py")


def _executable(clone: Path) -> None:
    (clone / "app.py").chmod(0o755)


def _bidi(clone: Path) -> None:
    (clone / "tool\u202e.py").write_text("x = 1\n")


def _undo(*names: str) -> Callable[[Path], None]:
    def undo(clone: Path) -> None:
        for name in names:
            path = clone / name
            if path.is_symlink() or not path.exists():
                path.unlink(missing_ok=True)
            elif name == "app.py":
                path.chmod(0o644)
            else:
                path.unlink()
    return undo


# The builder works in a clone of the attempt and its emission is derived from the clone's
# diff (`workspace.py`), so a malformed block cannot reach materialise from design mode any
# more. What can: a change the blocks cannot carry (refused by the collect step), a
# workspace with no change, and a file whose text materialise itself refuses.
@pytest.mark.parametrize("emission, first, retry, reason", [
    (render({"tool.py": "x = 'long'  # ... (line truncated to 2000 chars)\n"}), lambda _: None,
     lambda _: None, "the read tool's note that it cut"),
    (_edit("app.py", ("print('not there')\n", "print('x')\n")), lambda _: None, lambda _: None,
     "has no change from the last attempt"),
    (render({"tool.py": "print('built')\n"}), _binary, _undo("logo.png"), "logo.png is a binary file"),
    (render({"tool.py": "print('built')\n"}), _symlink, _undo("link.py"), "link.py is a symlink"),
    (render({"tool.py": "print('built')\n"}), _executable, _undo("app.py"),
     "app.py changes its file mode (100644 to 100755)"),
    # A name a terminal or a review misshows is refused by name, as an emitted block's is
    # (`repo_change.unprintable_refusal`, dm-208), and by collect itself, before materialise
    # sees it (dm-278): the reason carries collect's own prefix.
    (render({"tool.py": "print('built')\n"}), _bidi, _undo("tool\u202e.py"),
     "cannot be carried as text files, so nothing of it was applied: the emitted path "
     "'tool\\u202e.py' holds U+202E RIGHT-TO-LEFT OVERRIDE"),
], ids=["truncated-read", "no-change", "binary", "symlink", "mode-change", "bidi-path"])
def test_an_emission_materialise_cannot_apply_goes_back_to_emit_with_the_reason(repo, tmp_path, emission,
                                                                               first, retry, reason):
    """It once ended the run (#330 E5); now the retry is told why, and nothing of it is committed."""
    model = _WorkspaceBuilder(_answers(emission, render({"tool.py": "print('built')\n"})), first, retry)
    result = _design(tmp_path / "run", model, target=str(repo))
    assert result.exit_ok, result.error
    retry = [prompt for prompt in model.prompts if "Write the artifact `" in prompt][1]
    assert reason in retry
    payload = _report(tmp_path / "run")
    assert git(repo, "log", "--format=%s", f"{payload['base']}..{payload['branch']}").splitlines() == [
        f"{NAME}: attempt 1"]


def test_emissions_that_never_apply_exhaust_the_gate_iteration_limit(repo, tmp_path):
    """Refused emissions trigger the consecutive-refusal bound (default 3) before the gate's
    max_iterations (8). The refusal counter in materialise raises BuildLoopStop."""
    bad = _edit("app.py", ("print('not there')\n", "x\n"))
    # 4 bad emissions: the refusal bound (default 3) fires on the third consecutive refusal.
    result = _design(tmp_path / "run", FakeModel(_answers(*([bad] * 4))), target=str(repo))
    assert not result.exit_ok
    assert "emissions refused" in str(result.error) or "exceeded max_iterations" in str(result.error)
    assert not result.facts.get("attempts_used"), "refused emissions must not spend attempts"
    # A person reads result.facts["validation"] via `refactory status`; it holds the refusal.
    assert "materialise could not apply emit.txt" in result.facts.get("validation", "")


def test_a_no_change_emission_is_refused_and_does_not_spend_an_attempt(repo, tmp_path, monkeypatch):
    """An emission that changes nothing is refused at materialise and takes the same path as any
    refused emission: back to emit via Verdict('reloop'), no attempt spent.
    """
    # Phase 1 — materialise refuses the no-change emission
    run = tmp_path / "run"
    ctx = nn.RunContext(run, model=FakeModel([]),
                        task={"name": NAME, "brief": "a tool", "check": "true", "target": str(repo)})
    ctx.facts.update(design_ops.prepare_design(ctx))
    emit_text = render({"tool.py": "print('built')\n"})
    ctx.write_text("emit.txt", emit_text)
    result1 = repo_change.materialise(ctx)
    assert "emission_refused" not in result1
    head_after_first = worktree.recorded(run).head

    ctx.write_text("emit.txt", emit_text)
    result2 = repo_change.materialise(ctx)
    assert "emission_refused" in result2
    assert "changed nothing" in result2["emission_refused"]
    assert worktree.recorded(run).head == head_after_first

    # Phase 2 — check_artifact returns Verdict("reloop") without spending attempts
    monkeypatch.setattr(repo_change, "refused", lambda ctx: "the emission changed nothing: every file and edit in it is already in the last attempt")
    ctx2 = nn.RunContext(tmp_path / "run2", facts={"check": "python tool.py"},
                         knobs={"attempts": 3})
    ctx2._declared = ("check.txt",)
    ctx2._declared_facts = ("validation", "attempts_used")
    ctx2.enforce_writes = False
    result = design_ops.check_artifact(ctx2)
    assert isinstance(result, Verdict)
    assert str(result) == "reloop"
    assert "materialise could not apply" in result.reason
    assert "changed nothing" in result.reason
    assert not ctx2.facts.get("attempts_used"), "refused emission must not spend an attempt"
    # A person reads ctx.facts["validation"] via `refactory status`; it holds the refusal.
    assert "materialise could not apply emit.txt" in ctx2.facts.get("validation", "")


def test_the_same_edits_given_again_are_recognised_as_already_applied(repo, tmp_path):
    """The same edits given again are recognised as already applied: the refusal says the emission
    changed nothing, not that an edit's text is missing.  The branch holds one attempt per answer
    that changed something."""
    _commit(repo, "old.py", "x = 1\n")
    _commit(repo, "notes.py", "a = 1\nb = 2\nc = 3\n")
    tool_fail = ("import os, pathlib\n"
                 "marker = pathlib.Path(os.environ['HOME']) / 'second-attempt'\n"
                 "if not marker.exists():\n"
                 "    marker.write_text('')\n"
                 "    raise SystemExit('the first attempt is refused on purpose')\n")
    # A removal leaves nothing that shows it happened, so only the record of the emission
    # can tell this one is a repeat.
    emission = (render({"tool.py": tool_fail}) + _edit("app.py", ("print('app')\n", "print('edited')\n"))
                + _edit("notes.py", ("b = 2\n", "")) + "===DELETE: old.py===\n")
    # The second identical emission is refused ("changed nothing"), so a different
    # third emission must be given for the run to succeed.
    different = render({"tool.py": tool_fail, "extra.py": "y = 1\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emission, emission, different)), _mode(check_attempts=3),
                     target=str(repo))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    # First attempt commits, second is refused (changed nothing), third commits (adds extra.py).
    assert git(repo, "log", "--format=%s", f"{payload['base']}..{payload['branch']}").splitlines() == [
        f"{NAME}: attempt 2", f"{NAME}: attempt 1"]
    assert git(repo, "show", f"{payload['branch']}:app.py") == "print('edited')"
    assert git(repo, "show", f"{payload['branch']}:notes.py") == "a = 1\nc = 3"
    assert "extra.py" in git(repo, "ls-tree", "--name-only", payload["branch"]).split()


def test_an_empty_emission_is_refused_and_the_worktree_is_unchanged(repo, tmp_path):
    """An emission identical to the last attempt is refused at materialise, not committed."""
    run = tmp_path / "run"
    tool_code = "print('built')\n"
    emit_text = render({"tool.py": tool_code})

    # First attempt: set up worktree and commit
    ctx = nn.RunContext(run, model=FakeModel([]),
                        task={"name": NAME, "brief": "a tool", "check": "true", "target": str(repo)})
    ctx.facts.update(design_ops.prepare_design(ctx))
    ctx.write_text("emit.txt", emit_text)
    result1 = repo_change.materialise(ctx)
    assert "emission_refused" not in result1
    head_after_first = worktree.recorded(run).head

    # Second attempt: identical emission
    ctx.write_text("emit.txt", emit_text)
    result2 = repo_change.materialise(ctx)
    assert "emission_refused" in result2
    assert "changed nothing" in result2["emission_refused"]
    assert worktree.recorded(run).head == head_after_first


def test_an_edit_sent_again_is_skipped_rather_than_applied_twice(repo, tmp_path, capsys):
    """Models re-send an earlier attempt's edits beside new work; an insertion must not land twice.

    Design's builder works in a clone of the attempt, which already holds the insertion,
    so the retry's change is only what is new: the emission derived from it never re-sends
    app.py. Materialise's own repeat handling, for a mode that still writes its blocks,
    is `test_an_edit_whose_result_is_already_in_place_is_counted_as_applied`'s.
    """
    tool = ("import os, pathlib\n"
            "marker = pathlib.Path(os.environ['HOME']) / 'second-attempt'\n"
            "if not marker.exists():\n"
            "    marker.write_text('')\n"
            "    raise SystemExit('the first attempt is refused on purpose')\n")
    insert = _edit("app.py", ("print('app')\n", "print('app')\nprint('inserted')\n"))
    first = render({"tool.py": tool}) + insert
    second = render({"tool.py": tool, "extra.py": "x = 1\n"}) + insert
    result = _design(tmp_path / "run", FakeModel(_answers(first, second)), target=str(repo))
    assert result.exit_ok, result.error
    branch = _report(tmp_path / "run")["branch"]
    assert git(repo, "show", f"{branch}:app.py") == "print('app')\nprint('inserted')"
    assert (tmp_path / "run" / "emit.txt").read_text() == "===FILE: extra.py===\nx = 1\n"


@pytest.mark.parametrize("text, pair", [
    ("a\nnew\nb\n", ("a\n", "a\nnew\n")),
    ("new\na\nb\n", ("a\n", "new\na\n")),
    ("a\nx = 2\nb\n", ("x = 1\n", "x = 2\n")),
], ids=["inserted-after", "inserted-before", "changed"])
def test_an_edit_whose_result_is_already_in_place_is_counted_as_applied(text, pair):
    assert repo_change._edited(Path("a.py"), text, [pair], "git show HEAD:a.py") == (text, 1)


@pytest.mark.parametrize("pair", [("x\n", ""), ("x\n", "\n")], ids=["removed", "blanked"])
def test_an_edit_that_removes_lines_is_never_taken_as_already_applied(pair):
    """Nothing shows a removal happened -- a blank line proves nothing -- so it is refused, not skipped."""
    with pytest.raises(worktree.AttemptRefused, match="not in the file"):
        repo_change._edited(Path("a.py"), "a\n\nb\n", [pair], "git show HEAD:a.py")


def test_without_a_worktree_a_run_takes_whole_files_only(repo, tmp_path, monkeypatch):
    """Without a worktree the builder works in a folder of its own, and its files are the artifact.

    The builder no longer writes blocks, so it is told where it works rather than which
    blocks a run without a worktree takes; a deletion, which such a run cannot make, is
    refused by the collect step instead of by materialise.
    """
    monkeypatch.setenv("REFACTORY_PROJECT", str(repo))
    model = FakeModel(_answers(_edit("tool.py", ("x\n", "y\n")), render({"tool.py": "print('built')\n"})))
    result = _design(tmp_path / "run", model, _mode(worktree=False))
    assert result.exit_ok, result.error
    emits = [prompt for prompt in model.prompts if "Write the artifact `" in prompt]
    folder = tmp_path / "run" / ".refactory" / "workspaces" / "emit" / "repo"
    assert f"Your workspace is {folder}" in emits[0] and "===EDIT:" not in emits[0]
    assert "has no change from the last attempt" in emits[1]
    assert (Path(result.facts["artifact"]) / "tool.py").read_text() == "print('built')\n"


@pytest.mark.parametrize("answer, reason", [
    ("print('no blocks')\n", "holds no ===FILE:"),
    ("===EDIT: a path with spaces===\n", "malformed block header"),
    ("===FILE: a.py===\nx\n" + _edit("a.py", ("x\n", "y\n")), "more than one block.*===FILE:"),
    ("===DELETE: a.py===\nleftover\n", "a deletion takes none"),
    (f"===EDIT: a.py===\n{repo_change.SEARCH}\nx\n{repo_change.DIVIDER}\ny\n", "missing its"),
    (_edit("a.py", ("\n", "y\n")), "nothing to search for"),
    # A builder that writes its own blocks (create, edit, fast-create) can still send this.
    ("===EDIT: a.py===\nprint('no pairs')\n", "has text outside its"),
], ids=["no-blocks", "bad-header", "two-blocks", "delete-with-text", "unclosed-edit", "empty-search",
        "text-outside-pairs"])
def test_an_emission_whose_blocks_are_malformed_is_refused_with_the_reason(answer, reason):
    with pytest.raises(worktree.AttemptRefused, match=reason):
        repo_change._emitted(answer, edits=True)


def test_an_edit_whose_text_is_in_its_file_twice_is_refused_rather_than_guessed():
    with pytest.raises(worktree.AttemptRefused, match="appears 2 times"):
        repo_change._edited(Path("a.py"), "x = 1\nx = 1\n", [("x = 1\n", "x = 3\n")], "git show HEAD:a.py")


def test_an_edit_whose_lines_hold_a_divider_is_refused_rather_than_cut_short():
    """A reStructuredText title's underline is a line of `=======`, which once ended the search early."""
    with pytest.raises(worktree.AttemptRefused, match="unclear"):
        repo_change._emitted(_edit("README.rst", ("License\n=======\n", "Licence\n=======\n")), edits=True)
    with pytest.raises(worktree.AttemptRefused, match="missing; put one"):
        repo_change._emitted(f"===EDIT: a.py===\n{repo_change.SEARCH}\nx\n{repo_change.REPLACE}\n", edits=True)


@pytest.mark.parametrize("text, pairs, edited", [
    ("max = 1\nx = 1\n", [("\nx = 1\n", "\nx = 2\n")], "max = 1\nx = 2\n"),
    ("a\r\nb\r\n", [("b\n", "c\n")], "a\r\nc\r\n"),
    ("a\nb", [("b\n", "c\n")], "a\nc"),
    ("a\nb\nc\n", [("b\n", "")], "a\nc\n"),
], ids=["whole-line-with-context", "crlf-kept", "no-final-newline-kept", "lines-removed"])
def test_an_edit_matches_whole_lines_as_the_read_tool_shows_them_and_keeps_the_files_endings(text, pairs, edited):
    """The read tool strips `\\r` and ends every line, so that is the text a search is copied from."""
    assert repo_change._edited(Path("a.py"), text, pairs, "git show HEAD:a.py") == (edited, 0)


def test_a_short_search_inside_another_line_is_refused_as_appearing_twice():
    """'x = 1\\n' occurs both in 'max = 1\\n' and 'x = 1\\n'; the safe outcome is a refusal."""
    with pytest.raises(worktree.AttemptRefused, match="appears 2 times"):
        repo_change._edited(
            Path("a.py"), "max = 1\nx = 1\n",
            [("x = 1\n", "x = 2\n")], "git show HEAD:a.py")


def test_a_mid_line_search_applies_when_unique():
    """A search that starts inside a line, not at position 0 or after a newline, matches when unique."""
    result, skipped = repo_change._edited(
        Path("a.py"), "hello world\n", [("world\n", "earth\n")], "git show HEAD:a.py")
    assert result == "hello earth\n" and skipped == 0


def test_a_search_inside_a_line_longer_than_2000_characters_applies():
    """An edit can reach text past the 2000-char point the read tool truncates at."""
    line = "x" * 2500 + "MARKER" + "y" * 100 + "\n"
    result, skipped = repo_change._edited(
        Path("a.py"), line, [("MARKER", "REPLACED")], "git show HEAD:a.py")
    assert result == "x" * 2500 + "REPLACED" + "y" * 100 + "\n" and skipped == 0


def test_a_search_without_leading_spaces_applies_when_unique():
    """A search that omits the line's leading whitespace matches when the content is unique."""
    result, skipped = repo_change._edited(
        Path("a.py"), "    def foo():\n        pass\n",
        [("def foo():\n        pass\n", "def bar():\n        pass\n")],
        "git show HEAD:a.py")
    assert result == "    def bar():\n        pass\n" and skipped == 0


def test_a_substring_that_appears_twice_is_refused():
    """A search text appearing twice as a substring is refused, even when neither occurrence starts a line."""
    with pytest.raises(worktree.AttemptRefused, match="appears 2 times"):
        repo_change._edited(
            Path("a.py"), "abXYZcd\nefXYZgh\n",
            [("XYZ", "NEW")], "git show HEAD:a.py")


def test_an_absent_search_with_short_replacement_mid_line_is_refused_not_skipped():
    """An edit whose search text is absent and whose short replacement appears once mid-line is refused."""
    with pytest.raises(worktree.AttemptRefused, match="is not in the file"):
        repo_change._edited(
            Path("a.py"), "abREPLcd\n",
            [("GONE\n", "REPL")], "git show HEAD:a.py")


def test_a_mid_line_search_is_not_falsely_skipped_when_replacement_starts_a_line():
    """An edit whose search is mid-line applies even when the replacement starts another line."""
    result, skipped = repo_change._edited(
        Path("a.py"),
        "def f():\n    return compute(a)\ncompute(b)\n",
        [("compute(a)", "compute(b)")],
        "git show HEAD:a.py")
    assert result == "def f():\n    return compute(b)\ncompute(b)\n" and skipped == 0


def _applied(text: str, answer: str, name: str = "models.md") -> tuple[str, int]:
    """An emission's edits to one file, parsed as materialise parses them, then applied to ``text``."""
    _files, edits, _deletes = repo_change._emitted(answer, edits=True)
    return repo_change._edited(Path(name), text, edits[Path(name)], f"git show HEAD:{name}")


#: models.md keeps each paragraph on one line; b13b's refused edit searched mid-paragraph.
PARAGRAPH = ("the refused emission does not count against `check_attempts`; the gate's own iteration "
             "limit still bounds retries. An emission that changes nothing is refused back to emit.\n")
SENTENCE = "the gate's own iteration limit still bounds retries."
ADDED = " A CEO redo keeps the count."


def test_an_emitted_edit_that_stops_inside_a_line_applies():
    """The parser ends a search with a newline; a sentence inside a one-line paragraph is still found."""
    answer = _edit("models.md", (SENTENCE + "\n", SENTENCE + ADDED + "\n"))
    result, skipped = _applied(PARAGRAPH, answer)
    assert result == PARAGRAPH.replace(SENTENCE, SENTENCE + ADDED) and skipped == 0


def test_an_emitted_mid_line_insertion_sent_again_is_skipped_not_applied_twice():
    answer = _edit("models.md", (SENTENCE + "\n", SENTENCE + ADDED + "\n"))
    once, _ = _applied(PARAGRAPH, answer)
    assert _applied(once, answer) == (once, 1)


def test_an_emitted_mid_line_search_found_twice_is_refused():
    with pytest.raises(worktree.AttemptRefused, match="appears 2 times"):
        _applied("x. y.\nx. z.\n", _edit("models.md", ("x.\n", "w.\n")))


@pytest.mark.parametrize("pair, edited", [
    (("b\n", "B\n"), "a\nB\nc\n"),
    (("b\n", ""), "a\nc\n"),
], ids=["whole-line", "line-removed"])
def test_emitted_whole_line_edits_still_match_whole_lines(pair, edited):
    """A search that ends at a line's end keeps its newline, so a removed line leaves no blank."""
    assert _applied("a\nb\nc\n", _edit("models.md", pair)) == (edited, 0)


def test_multiple_edit_blocks_for_the_same_file_are_merged():
    """Several ===EDIT: blocks naming the same file are applied in order, as if one block."""
    answer = (_edit("a.py", ("aaa\n", "AAA\n"))
              + _edit("a.py", ("bbb\n", "BBB\n")))
    _, changed, _ = repo_change._emitted(answer, edits=True)
    assert list(changed.keys()) == [Path("a.py")]
    assert changed[Path("a.py")] == [("aaa\n", "AAA\n"), ("bbb\n", "BBB\n")]


def test_two_file_blocks_for_one_file_are_refused():
    """Two ===FILE: blocks for the same file are refused, saying which."""
    with pytest.raises(worktree.AttemptRefused, match="more than one block.*===FILE:"):
        repo_change._emitted("===FILE: a.py===\nx\n===FILE: a.py===\ny\n", edits=True)


def test_the_read_tools_paging_note_is_refused_too():
    """What a model retyping a long file is likeliest to copy: the note where the read stopped."""
    with pytest.raises(worktree.AttemptRefused, match="cut text short"):
        repo_change._emitted(render({"a.py": "x = 1\n(Showing lines 1-2000 of 2637. Use offset=2001 to continue.)\n"}),
                            edits=True)


def test_a_worktree_a_prune_removed_is_added_back_before_an_edit_applies(repo, tmp_path):
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "abcdef123456", run)
    git(repo, "worktree", "remove", "--force", found.path)
    ctx = nn.RunContext(run, model=FakeModel([]), task={})
    ctx.facts.update({"name": NAME})
    ctx.write_text("emit.txt", _edit("app.py", ("print('app')\n", "print('edited')\n")))
    facts = repo_change.materialise(ctx)
    assert "emission_refused" not in facts and facts["files"] == ["app.py"]
    assert (Path(found.path) / "app.py").read_text() == "print('edited')\n"


def test_on_the_spines_route_a_refusal_reaches_the_check_and_a_later_attempt_clears_it(repo, tmp_path, monkeypatch):
    """The spine runs each op in its own process; the refusal has to survive in facts.json."""
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef654321", run)
    lowering.write_declarations(DesignMode().state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": NAME, "brief": "a tool", "check": "true"}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 3}))
    (run / "emit.txt").write_text(_edit("app.py", ("print('not there')\n", "x\n")))
    node_runner.run_node(f"{repo_change.OPS}:materialise", workdir=run, node_id="materialise")
    verdict = node_runner.run_node(f"{design_ops.__name__}:check_artifact", workdir=run, node_id="check")
    assert verdict == "reloop" and "not in the file the edit applies to" in verdict.reason
    (run / "emit.txt").write_text(render({"tool.py": "print(1)\n"}))
    node_runner.run_node(f"{repo_change.OPS}:materialise", workdir=run, node_id="materialise")
    assert json.loads((run / "facts.json").read_text())["emission_refused"] == ""


def test_a_run_whose_graph_predates_the_refusal_fact_still_materialises(repo, tmp_path):
    """A spine run started before this fact existed has it in no declaration; succeeding must not need it."""
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef112233", run)
    lowering.write_declarations(DesignMode().state_dict(), run)
    declarations = json.loads((run / lowering.DECLARATIONS).read_text())
    declarations["materialise"]["produces"].remove("emission_refused")
    (run / lowering.DECLARATIONS).write_text(json.dumps(declarations))
    (run / "facts.json").write_text(json.dumps({"name": NAME, "check": "python tool.py"}))
    (run / "emit.txt").write_text(render({"tool.py": "print(1)\n"}))
    node_runner.run_node(f"{repo_change.OPS}:materialise", workdir=run, node_id="materialise")
    assert json.loads((run / "facts.json").read_text())["files"] == ["tool.py"]


# ── the boundaries a run keeps ─────────────────────────────────────────────
class _EditingModel(FakeModel):
    """The #319 worker: edits a checkout directly, then answers that it is already done."""

    def __init__(self, answers: list[str], edit: Callable[[], Path]) -> None:
        super().__init__(answers)
        self.edit = edit

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
        if "Write the artifact `" in prompt:
            (self.edit() / "app.py").write_text("print('edited in place')\n")
        return answer


def test_an_emit_that_edits_the_worktree_is_refused(repo, tmp_path):
    """The worktree is the run's own, but only materialise writes it (#319)."""
    run = tmp_path / "run"
    edit = lambda: Path(worktree.recorded(run).path)
    result = _design(run, _EditingModel(
        _answers(render({"tool.py": "print('built')\n"})), edit),
        target=str(repo))
    assert result.exit_ok is False
    assert "changed while agent 'emit' ran" in (result.error or "")
    found = worktree.recorded(run)
    assert f"in {found.path}, " in (result.error or "")
    assert found.head == found.base, "materialise never ran"


def test_an_emit_during_which_the_target_moves_warns_but_succeeds(
        repo, tmp_path, capsys):
    """An emit that writes into the target is now refused and restored (dm-159).

    The _EditingModel writes app.py into the target during emit.  emit is
    output-only, so strayed() moves target_changes from warn to fail.  The
    target is restored and the set-aside copy of app.py exists.
    """
    run = tmp_path / "run"
    result = _design(run, _EditingModel(
        _answers(render({"tool.py": "print('built')\n"})), lambda: repo),
        target=str(repo))
    assert result.exit_ok is False
    assert "changed while agent" in (result.error or "")
    # The target is restored — app.py has its committed content.
    assert (repo / "app.py").read_text() == "print('app')\n"
    # The set-aside copy exists under agent-delivery.
    from refactory_dsh import agent_delivery
    found_set_aside = list((run / agent_delivery.ROOT).rglob("target-set-aside/app.py"))
    assert found_set_aside, "the set-aside copy of app.py must exist"


def test_a_refused_emit_is_redone_from_a_fresh_emit_txt(repo, tmp_path):
    """Left in the run, the refused emit.txt read to the redo's worker as the work done."""
    run = tmp_path / "run"
    refused = _design(run, _EditingModel(_answers("All changes are already applied."),
                                         lambda: Path(worktree.recorded(run).path)), target=str(repo))
    assert "changed while agent 'emit' ran" in (refused.error or "")
    git(worktree.recorded(run).path, "checkout", "--", "app.py")

    redone = DesignMode.resume(run, model=StaleEmitModel([render({"tool.py": "print('redone')\n"}),
                                                          *REVIEW_ANSWERS[:3],
                                                          VERIFICATION_ANSWER,
                                                          REVIEW_ANSWERS[3]]))
    assert redone.error is None
    # The redo's emission is derived from its workspace's diff: the file it wrote, whole.
    assert (run / "emit.txt").read_text() == "===FILE: tool.py===\nprint('redone')\n"
    assert git(repo, "show", f"{_report(run)['branch']}:tool.py") == "print('redone')"


class _HookChangingModel(FakeModel):
    """An agent that rewrites a shared hook while a planning step runs (#347)."""

    def __init__(self, answers: list[str], target_fn) -> None:
        super().__init__(answers)
        self._target_fn = target_fn

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
        if "You are the research director" in prompt:
            import subprocess
            target = self._target_fn()
            hooks = Path(subprocess.run(
                ['git', '-C', str(target), 'rev-parse', '--path-format=absolute',
                 '--git-common-dir'], capture_output=True, text=True,
                check=True).stdout.strip()) / 'hooks'
            hooks.mkdir(parents=True, exist_ok=True)
            (hooks / 'pre-commit').write_text('#!/bin/sh\nexit 0\n')
        return answer


def test_materialise_refusal_clears_emit_txt_before_redispatch(repo, tmp_path):
    """After materialise refuses an emission, the stale emit.txt is gone before the next emit runs."""
    bad_emission = _edit("app.py", ("print('not there')\n", "print('x')\n"))
    good_emission = render({"tool.py": "print('built')\n"})
    model = StaleEmitModel(_answers(bad_emission, good_emission))
    result = _design(tmp_path / "run", model, target=str(repo))
    assert result.exit_ok, result.error
    # StaleEmitModel checks: if emit.txt existed when the second emit ran,
    # it would have returned "emit.txt already contains the correct content"
    # instead of the good_emission answer. The run succeeding proves emit.txt
    # was gone.


def test_check_failure_clears_emit_txt_before_redispatch(repo, tmp_path):
    """After a failed check, the stale emit.txt is gone before the next emit runs."""
    failing = render({"tool.py": "raise SystemExit(1)\n"})
    fixed = render({"tool.py": "print('fixed')\n"})
    model = StaleEmitModel(_answers(failing, fixed))
    result = _design(tmp_path / "run", model, target=str(repo))
    assert result.exit_ok, result.error
    # StaleEmitModel would have intercepted the second emit if emit.txt survived.
    # The run succeeding with the fixed code proves emit.txt was moved.
    payload = _report(tmp_path / "run")
    branch = payload["branch"]
    assert git(Path(payload["target"]), "show", f"{branch}:tool.py") == "print('fixed')"


def test_a_planning_step_that_changes_shared_hooks_fails_the_step(repo, tmp_path):
    """End-to-end: full design run, planning step changes hooks -> run fails (#347)."""
    model = _HookChangingModel(
        _answers(render({"tool.py": "print('built')\n"})),
        lambda: repo,
    )
    result = _design(tmp_path / "run", model, target=str(repo))
    assert result.exit_ok is False
    assert "changed shared git state" in (result.error or "")
    assert "research_director" in (result.error or "")


def test_a_check_in_the_worktree_sees_none_of_refactorys_environment(repo, tmp_path, monkeypatch):
    """`REFACTORY_*` and the launcher's PYTHONPATH once failed a target's own tests (#330 E1)."""
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "launcher"))
    monkeypatch.setenv("REFACTORY_PROBE", "1")
    tool = ("import os, pathlib\n"
            "leaked = sorted(n for n in os.environ if n == 'PYTHONPATH' or n.startswith(('REFACTORY_', 'DSH_')))\n"
            "assert not leaked, leaked\n"
            "assert not pathlib.Path(os.environ['HOME']).resolve().is_relative_to(pathlib.Path.cwd().resolve())\n"
            "print(os.environ['HOME'])\n")
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": tool}))), target=str(repo))
    assert result.exit_ok, result.error
    evidence = json.loads((tmp_path / "run" / "authoring-evidence.json").read_text())
    # The check runs beside the reviews, in the check copy, with the copy's own HOME.
    assert evidence["validation"]["stdout"].strip() == str(tmp_path / "run" / ".refactory" / "check-copy-home")


def test_a_check_in_the_worktree_is_bounded_by_its_timeout_not_by_memory(repo, tmp_path):
    """A Node check that starts worker threads, as vitest does, failed under a 2 GB address-space cap (#325)."""
    import resource

    tool = ("import json, resource\n"
            "print(json.dumps([resource.getrlimit(resource.RLIMIT_AS)[0], "
            "resource.getrlimit(resource.RLIMIT_CPU)[0]]))\n")
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": tool}))), _mode(check_timeout_s=30),
                     target=str(repo))
    assert result.exit_ok, result.error
    evidence = json.loads((tmp_path / "run" / "authoring-evidence.json").read_text())
    assert json.loads(evidence["validation"]["stdout"]) == [resource.getrlimit(resource.RLIMIT_AS)[0],
                                                            30 * (os.cpu_count() or 1)]


def test_a_check_uses_the_environment_its_worktrees_manifests_install(repo, tmp_path, monkeypatch):
    """The worktree's interpreter, not refactory's, even though refactory's is on the launcher's PATH."""
    stub(tmp_path / "stubs", "uv", "mkdir -p .venv/bin\n"
                                   "printf '#!/bin/sh\\necho the-worktree-python\\n' > .venv/bin/python\n"
                                   "chmod +x .venv/bin/python")
    monkeypatch.setenv("PATH", f"{tmp_path / 'stubs'}{os.pathsep}{os.environ['PATH']}")
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "-q", "-m", "manifest")
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": "print('refactory python')\n"}))),
                     target=str(repo))
    assert result.exit_ok, result.error
    evidence = json.loads((tmp_path / "run" / "authoring-evidence.json").read_text())
    assert evidence["validation"]["stdout"].strip() == "the-worktree-python"


def test_the_setup_command_runs_in_the_worktree_before_each_check(repo, tmp_path):
    tool = "import pathlib\nassert pathlib.Path('built.txt').read_text() == 'built\\n'\n"
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": tool}))),
                     _mode(setup="echo built > built.txt"), target=str(repo))
    assert result.exit_ok, result.error


def test_a_missing_installer_stops_the_run_without_spending_its_repairs(repo, tmp_path, monkeypatch):
    """No code change can install uv; the run says so instead of asking emit to try again."""
    only_git = tmp_path / "only-git"
    only_git.mkdir()
    (only_git / "git").symlink_to(next(Path(entry) / "git" for entry in os.environ["PATH"].split(os.pathsep)
                                       if (Path(entry) / "git").is_file()))
    monkeypatch.setenv("PATH", str(only_git))
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "-q", "-m", "manifest")
    emit = render({"tool.py": "print('built')\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emit, emit)), target=str(repo))
    assert result.exit_ok is False
    assert "UNAVAILABLE: uv is not installed" in (result.error or "")
    found = worktree.recorded(tmp_path / "run")
    assert git(repo, "rev-list", "--count", f"{found.base}..{found.branch}") == "1", "no repair was attempted"


def test_a_setup_command_that_fails_fails_the_check_and_says_why(tmp_path):
    emit1 = render({"tool.py": "print('built')\n"})
    emit2 = render({"tool.py": "print('built')\n", "note.txt": "r1\n"})
    emit3 = render({"tool.py": "print('built')\n", "note.txt": "r2\n"})
    failed_setup = _design(tmp_path / "setup", FakeModel(_answers(emit1, emit2, emit3)), _mode(setup="exit 3"),
                           target=str(git_repo(tmp_path / "plain")))
    assert failed_setup.exit_ok is False
    assert "the setup command `exit 3` failed" in (failed_setup.error or "")


def test_a_generate_command_amends_tracked_file_changes_into_the_attempt(repo, tmp_path):
    """Whatever a generate command changes in tracked files is committed with the attempt."""
    _commit(repo, "generated.txt", "old\n")
    emit = render({"tool.py": "print('built')\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emit)),
                     _mode(generate="echo new > generated.txt"), target=str(repo))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert "generated.txt" in git(repo, "ls-tree", "--name-only", payload["branch"]).splitlines()
    assert git(repo, "show", f"{payload['branch']}:generated.txt") == "new"
    assert payload["head"] == git(repo, "rev-parse", payload["branch"]) == result.facts["commit"]
    assert "generated.txt" in (tmp_path / "run" / "changes.diff").read_text()
    assert "generated.txt" in result.facts["files"]
    # The amendment amends the attempt, not a separate commit.
    assert git(repo, "rev-list", "--count", f"{payload['base']}..{payload['branch']}") == "1"


def test_a_generate_command_that_changes_nothing_adds_no_commit(repo, tmp_path):
    """A generate that touches nothing leaves the attempt as the one commit on the branch."""
    emit = render({"tool.py": "print('built')\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emit)),
                     _mode(generate="true"), target=str(repo))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert git(repo, "rev-list", "--count", f"{payload['base']}..{payload['branch']}") == "1"


def test_a_generate_command_that_fails_fails_the_check_and_says_why(tmp_path):
    emit1 = render({"tool.py": "print('built')\n"})
    emit2 = render({"tool.py": "print('built')\n", "note.txt": "r1\n"})
    emit3 = render({"tool.py": "print('built')\n", "note.txt": "r2\n"})
    failed_generate = _design(tmp_path / "run", FakeModel(_answers(emit1, emit2, emit3)),
                              _mode(generate="exit 3"),
                              target=str(git_repo(tmp_path / "plain")))
    assert failed_generate.exit_ok is False
    assert "the generate command `exit 3` failed" in (failed_generate.error or "")


def test_generate_runs_before_setup_so_setup_sees_generated_files(repo, tmp_path):
    """A setup that needs generated files passes because generate runs first.

    With the old order (setup, then generate) this fails: setup sees stale
    generated.txt before generate can update it.
    """
    _commit(repo, "source.txt", "v1\n")
    _commit(repo, "generated.txt", "v1\n")
    # The emit changes source.txt; generate copies it to generated.txt; setup diffs them.
    emit = render({"tool.py": "print('built')\n", "source.txt": "v2\n"})
    result = _design(
        tmp_path / "run", FakeModel(_answers(emit)),
        _mode(generate="cp source.txt generated.txt",
              setup="diff -q source.txt generated.txt"),
        target=str(repo),
    )
    assert result.exit_ok, result.error


def test_a_lockfile_the_install_writes_is_part_of_the_attempt_the_reviewers_see(repo, tmp_path, monkeypatch):
    """A new Node project has no lockfile: the one its install writes is committed, not refused."""
    stub(tmp_path / "bin", "npm", "echo '{\"lockfileVersion\": 3}' > package-lock.json")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    emit = render({"package.json": "{}\n", "tool.py": "print('built')\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emit)), target=str(repo))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert "package-lock.json" in git(repo, "ls-tree", "--name-only", payload["branch"]).splitlines()
    assert payload["head"] == git(repo, "rev-parse", payload["branch"]) == result.facts["commit"]
    assert "package-lock.json" in (tmp_path / "run" / "changes.diff").read_text()
    assert "package-lock.json" in result.facts["files"], "the files the branch delivers follow the amendment"


def test_a_relative_target_names_a_repository_beside_the_project_not_the_run_directory(repo, tmp_path,
                                                                                    monkeypatch):
    """A node starts in its run directory; `target=repo` from the GUI means the project's `repo`."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    result = _design(tmp_path / "runs" / "one", FakeModel(_answers(render({"tool.py": "print(1)\n"}))),
                     target="repo")
    assert result.exit_ok, result.error
    assert _report(tmp_path / "runs" / "one")["target"] == str(repo.resolve())


def test_a_base_for_a_project_that_does_not_exist_yet_is_refused_before_it_is_made(tmp_path):
    result = _design(tmp_path / "run", FakeModel(_answers()), base="v1")
    assert result.exit_ok is False
    assert "this run starts a new project, which has only its empty main" in (result.error or "")
    assert not (projects_dir() / NAME).exists()


def test_a_second_run_on_a_project_names_the_earlier_runs_branches(tmp_path, capsys):
    """A new project's main stays empty until #350 delivers, so the work is on the branches."""
    first = _design(tmp_path / "one", FakeModel(_answers(render({"tool.py": "print(1)\n"}))))
    assert first.exit_ok, first.error
    capsys.readouterr()
    second = _design(tmp_path / "two", FakeModel(_answers(render({"tool.py": "print(2)\n"}))))
    assert second.exit_ok, second.error
    warned = capsys.readouterr().err
    assert f"has earlier runs, whose work is on their branches, not on main: {_report(tmp_path / 'one')['branch']}" in warned


def test_an_emit_during_which_the_launcher_checkout_moves_warns_but_succeeds(
        repo, tmp_path, monkeypatch, capsys):
    """Workers start in the run directory inside that checkout; in a worktree
    run a change there is a warning, not a failure (#319)."""
    launcher = git_repo(tmp_path / "launcher")
    monkeypatch.setenv("REFACTORY_PROJECT", str(launcher))
    result = _design(
        tmp_path / "run",
        _EditingModel(_answers(render({"tool.py": "print(1)\n"})),
                      lambda: launcher),
        target=str(repo))
    assert result.exit_ok is True, result.error
    warned = capsys.readouterr().err
    assert f"in {launcher.resolve()}" in warned
    assert "files outside its worktree changed" in warned


def test_a_worktree_run_does_not_warn_that_the_persons_checkout_moved(repo, tmp_path, capsys):
    """A person's edit during a planning step is now refused and the target is restored (dm-159).

    research_director is output-only (in PLANNING_STEPS), so strayed() moves
    target_changes from warn to fail.  The refusal restores the target and the
    set-aside copy of app.py exists.
    """
    class Moving(FakeModel):
        def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
            if "You are the research director" in prompt:
                (repo / "app.py").write_text("print('the person kept working')\n")
            return super().complete(prompt, system=system, workdir=workdir, **workspace)

    result = _design(tmp_path / "run", Moving(_answers(render({"tool.py": "print(1)\n"}))), target=str(repo))
    assert result.exit_ok is False
    assert "changed while agent" in (result.error or "")
    # The target is restored — app.py has its committed content.
    assert (repo / "app.py").read_text() == "print('app')\n"
    # The set-aside copy exists under agent-delivery.
    from refactory_dsh import agent_delivery
    found_set_aside = list((tmp_path / "run" / agent_delivery.ROOT).rglob("target-set-aside/app.py"))
    assert found_set_aside, "the set-aside copy of app.py must exist"


def test_the_agents_are_told_the_worktree_and_the_reviewers_the_diff_and_the_wrapper(repo, tmp_path):
    model = FakeModel(_answers(render({"tool.py": "print('built')\n"})))
    result = _design(tmp_path / "run", model, target=str(repo))
    assert result.exit_ok, result.error
    found = worktree.recorded(tmp_path / "run")
    assert any(f"answerable by reading {found.path}" in prompt for prompt in model.prompts)
    # The builder works in a clone of the last attempt, and is told the worktree is not its to change.
    emit = next(prompt for prompt in model.prompts if "Write the artifact `" in prompt)
    clone = tmp_path / "run" / ".refactory" / "workspaces" / "emit" / "repo"
    assert f"Your workspace is {clone}: a clone of the run's worktree at its last attempt" in emit
    assert f"The run's worktree at {found.path} holds the attempts as committed; do not change it" in emit
    assert "the workspace's diff is your answer" in emit and "===EDIT: <relative path>===" not in emit
    reviews = [prompt for prompt in model.prompts if "do not infer it from your working directory" in prompt]
    assert reviews
    for prompt in reviews:
        assert f"git -C {found.path} diff {found.base[:12]}" in prompt
        assert str(tmp_path / "run" / worktree.WRAPPER) in prompt


# ── target guard tests (dm-159) ────────────────────────────────────────────


def test_target_write_with_uncommitted_edit_and_untracked_file_is_refused_and_restored(tmp_path):
    """Test 1: person's uncommitted edit + untracked file; agent overwrites; target restored."""
    from refactory_dsh import agent_delivery, target_guard, worktree
    import subprocess

    repo = git_repo(tmp_path / "target")
    # Person edits app.py (uncommitted) and creates notes.txt (untracked).
    (repo / "app.py").write_text("print('mine, not committed')\n")
    (repo / "notes.txt").write_text("untracked\n")
    before_content = {}
    for f in ("app.py", "notes.txt"):
        before_content[f] = (repo / f).read_bytes()

    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0001guard", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    # Agent overwrites app.py and deletes notes.txt in the target.
    (repo / "app.py").write_text("print('agent wrote here')\n")
    (repo / "notes.txt").unlink()
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")

    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")

    # Target is byte-identical to before.
    for f, expected in before_content.items():
        assert (repo / f).read_bytes() == expected, f"{f} not restored"

    # Receipt says failed.
    receipt = agent_delivery.receipts_in(run)["emit"]
    assert receipt["status"] == "failed"


def test_person_editing_during_node_target_restored(tmp_path):
    """Test 2: person edits during node; both agent's and person's concurrent edits restored."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0002conc", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    before_bytes = (repo / "app.py").read_bytes()
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    # Person edits app.py while the node runs.
    (repo / "app.py").write_text("print('person concurrent edit')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")

    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")

    # Target is byte-identical to its state before the node.
    assert (repo / "app.py").read_bytes() == before_bytes


def test_verifier_writing_to_target_is_refused(tmp_path):
    """Test 3: non-output-only agent (verifier) writing to target is refused."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0003veri", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    before_bytes = (repo / "app.py").read_bytes()
    agent_delivery.begin(run, "verifier", "Verify.", ["verification.json"])
    # Verifier writes stray into target.
    (repo / "stray.py").write_text("print('stray')\n")
    (run / "verification.json").write_text("{}\n")

    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "verifier")

    # Target is restored: app.py unchanged, stray.py gone.
    assert (repo / "app.py").read_bytes() == before_bytes
    assert not (repo / "stray.py").exists()


def test_self_targeting_worktree_output_only_node_not_refused(tmp_path):
    """Test 4: no false refusal for a self-targeting worktree with output-only writing its outputs."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    found = worktree.create(repo, "HEAD", "tg0004safe", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    receipt = agent_delivery.finish(run, "emit")
    assert receipt["status"] == "delivered"


def test_parallel_agents_stray_absent_from_target(tmp_path):
    """Test 8: parallel agents — stray.py absent from target, qa_director not refused."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0008para", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    # code_reviewer writes stray.py into the target.
    agent_delivery.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (repo / "stray.py").write_text("print('stray')\n")
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "code_reviewer")

    # stray.py is now set aside and absent from the target.
    assert not (repo / "stray.py").exists()

    # qa_director starts — its snapshot includes stray.py's absence as the current state.
    agent_delivery.begin(run, "qa_director", "QA.", ["reviews/qa.md"])
    (run / "reviews" / "qa.md").write_text("VERDICT: pass\n")
    # qa_director makes no target changes; the only diff is stray.py, which is
    # discounted because it was recorded as a stray at or after watched_at.
    receipt = agent_delivery.finish(run, "qa_director")
    assert receipt["status"] == "delivered"
    assert not (repo / "stray.py").exists()


def test_stale_stray_does_not_stop_restore(tmp_path):
    """s17: a stray recorded before this node's watch never stops this node's change from being restored."""
    from refactory_dsh import agent_delivery, target_guard, worktree
    import time

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0017stal", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    # First agent writes stray.py, is refused.
    agent_delivery.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (repo / "stray.py").write_text("print('first stray')\n")
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    import pytest
    with pytest.raises(ValueError):
        agent_delivery.finish(run, "code_reviewer")
    assert not (repo / "stray.py").exists()

    # A later agent overwrites the same path — the stale stray record must NOT stop the restore.
    time.sleep(0.01)  # ensure watched_at is later
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "stray.py").write_text("print('later overwrite')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")
    # stray.py was restored (set aside) — not left in the target.
    assert not (repo / "stray.py").exists()


def test_executable_mode_restored(tmp_path):
    """Test 10: executable file mode (+x) restored after refusal."""
    from refactory_dsh import agent_delivery, target_guard, worktree
    import os
    import subprocess

    repo = git_repo(tmp_path / "target")
    (repo / "run.sh").write_text("#!/bin/sh\necho hello\n")
    os.chmod(repo / "run.sh", 0o755)
    subprocess.run(["git", "-C", str(repo), "add", "run.sh"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "add script"], check=True, capture_output=True)

    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0010exec", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    # Agent edits the script and changes its mode.
    (repo / "run.sh").write_text("#!/bin/sh\necho hacked\n")
    os.chmod(repo / "run.sh", 0o644)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")

    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")

    # File mode is restored.
    assert os.stat(repo / "run.sh").st_mode & 0o755 == 0o755
    assert (repo / "run.sh").read_text() == "#!/bin/sh\necho hello\n"
    # git status should be clean.
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert status == ""


def test_begin_handles_untracked_directory(tmp_path):
    """Test 11: begin() handles an untracked directory without IsADirectoryError."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    # Create an untracked directory (e.g. a nested checkout).
    (repo / "nested").mkdir()
    (repo / "nested" / "file.txt").write_text("data\n")

    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0011dir0", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    # begin() must not raise IsADirectoryError.
    record = agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    assert record is not None


def test_emit_running_generator_with_contract_python_is_refused(tmp_path):
    """Test 9: emit runs a generator with the contract's python — refused, target byte-identical."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    before_bytes = (repo / "app.py").read_bytes()

    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0009genp", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    record = agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    prompt = record["prompt"]

    # The contract text warns that the python is for nested dispatch only.
    assert "nested dispatch only" in prompt

    # Simulate the emit builder running a generator with the contract's python,
    # which writes into the driver checkout (the target).  The generator modifies
    # a tracked file and adds an untracked one — the same effect as running
    # `python -m gen_discovery` with the contract interpreter when refactory_source
    # points at the target.
    (repo / "app.py").write_text("print('generator overwrote this')\n")
    (repo / "generated_data.py").write_text("DATA = {}\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")

    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")

    # Target is byte-identical: tracked file restored, untracked file removed.
    assert (repo / "app.py").read_bytes() == before_bytes
    assert not (repo / "generated_data.py").exists()


def test_head_moved_refusal_text(tmp_path):
    """s5: when HEAD moved during the node, the refusal says so."""
    from refactory_dsh import agent_delivery, target_guard, worktree
    import subprocess

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0005head", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    # Move HEAD in the target.
    (repo / "new.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "new.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "move head"], check=True, capture_output=True)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")

    import pytest
    with pytest.raises(ValueError, match="changed while agent") as exc_info:
        agent_delivery.finish(run, "emit")
    assert "HEAD moved" in str(exc_info.value)
    assert "commits are not rewritten" in str(exc_info.value)


def test_worktree_only_stray_no_could_not_be_restored(tmp_path, capsys):
    """s6c: a worktree-only stray (target untouched) does not say 'could not be restored'."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0006cwt0", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    # Only change the worktree, not the target.
    wt = Path(found.path)
    (wt / "extra.py").write_text("print('worktree only')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")

    import pytest
    with pytest.raises(ValueError) as exc_info:
        agent_delivery.finish(run, "emit")
    # The refusal should NOT say "could not be restored" when target is clean.
    assert "could not restore" not in str(exc_info.value).lower()


def test_snapshot_resaved_on_reused_dispatch(tmp_path):
    """The baseline is re-saved when a reused dispatch re-watches the target."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0012reus", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    # First dispatch with reuse.
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (repo / "app.py").write_text("print('dirty')\n")
    (run / "emit.txt").write_text("Done.\n")
    import pytest
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")

    # Retry with reuse — the baseline must be re-saved.
    (repo / "app.py").write_text("print('person fixed it')\n")
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    # Now agent writes into the target again.
    (repo / "app.py").write_text("print('agent again')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")
    # The person's fixed content is restored, not the original committed content.
    assert (repo / "app.py").read_text() == "print('person fixed it')\n"


def test_contract_scope_text_survives_prompt_edit(tmp_path):
    """Test 7: the contract scope text is present even when the prompt drops 'Write nothing but emit.txt'."""
    from refactory_dsh import agent_delivery, target_guard, worktree

    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", "tg0007scope", run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))

    # Dispatch with a prompt that has no "Write nothing but emit.txt" but keeps placeholders.
    record = agent_delivery.begin(run, "emit", "Just build the tool. {{emit_scope}}", ["emit.txt"])
    prompt = record["prompt"]
    assert "Write only the paths in `outputs` below" in prompt
    assert f"Never change the target at {found.target}" in prompt


# ── what keeps working as before ───────────────────────────────────────────
def test_with_its_worktree_off_a_run_writes_the_artifact_directory_and_makes_no_branch(repo, tmp_path,
                                                                                    monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(repo))
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": "print('built')\n"}))),
                     _mode(worktree=False))
    assert result.exit_ok, result.error
    assert Path(_report(tmp_path / "run")["artifact"]) == (repo / repo_change.ARTIFACTS / NAME).resolve()
    assert git(repo, "branch", "--list", "refactory/*") == ""
    assert not worktree.root().exists() and projects.load()["projects"] == []


def test_with_its_worktree_off_a_run_refuses_what_only_a_worktree_can_use(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(repo))
    result = _design(tmp_path / "run", FakeModel(_answers()), _mode(worktree=False), target=str(repo))
    assert result.exit_ok is False
    assert "target need the run's worktree, which is off" in (result.error or "")
    # Following up a pull request is design's own (#350), and needs a worktree as much.
    result = _design(tmp_path / "again", FakeModel(_answers()), _mode(worktree=False), refine="acme/app#3")
    assert result.exit_ok is False
    assert "refine need the run's worktree, which is off" in (result.error or "")


def test_a_design_run_lowered_before_materialise_moved_still_finds_it():
    """A run waiting at its approval gate across an upgrade calls the name its graph was lowered with."""
    from refactory_dsh.nn.resolve import resolve

    assert resolve("refactory_dsh.design_ops:materialise_artifact") is repo_change.materialise


def test_a_run_without_a_worktree_never_inherits_one_an_earlier_run_left_in_its_directory(
        repo, tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(repo))
    run = tmp_path / "run"
    for knobs, target in (({}, str(repo)), ({"worktree": False}, "")):
        ctx = nn.RunContext(run, model=FakeModel([]), knobs=knobs,
                            task={"name": NAME, "brief": "a tool", "check": "python tool.py", "target": target})
        ctx.facts.update(design_ops.prepare_design(ctx))
    assert worktree.recorded(run) is None
    assert ctx.facts["worktree"] == repo_change.NO_WORKTREE


# ── design-document contract gate check ────────────────────────────────────────


def test_a_design_missing_a_contract_part_reloops_before_the_gate(tmp_path):
    """A contract-format design with an empty ## Changes reloops from gate_design_review
    (an empty optional section, such as Rule and sites, does not)."""
    import refactory_dsh.design_ops as ops

    design_text = (
        "# Make the gate check constraints\n\n"
        "## Summary\n\nSplit constraints into clauses.\n\n"
        "## Rule and sites\n\n"
        "## Docs\n\n"
        "- docs/models.md \u2014 add section\n\n"
        "## Changes\n\n"
    )
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "name": "b17_clauses",
                      "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", design_text)
    ctx.write_json("brief.json", {"name": "b17_clauses", "brief": "split constraints"})
    verdict = ops.gate_design_review(ctx)
    assert verdict == "reloop"
    assert "## Changes" in verdict.reason and "Rule and sites" not in verdict.reason


def test_a_rule_site_without_a_marker_is_advisory_at_the_gate(tmp_path):
    """A site the design lists without a [changed] or [left] marker is listed as advisory."""
    import refactory_dsh.design_ops as ops

    design_text = (
        "# Make the gate check constraints\n\n"
        "## Summary\n\nSplit constraints into clauses.\n\n"
        "## Ask mapping\n\n| Ask (quoted from brief) | Design section | Test |\n|---|---|---|\n"
        "| \"the brief\" | § Changes | test_tool.py |\n\n"
        "## Rule and sites\n\n"
        "- `authoring_evidence.py:154` \u2014 **[changed]**\n"
        "- `create_ops.py:1696` \u2014 the overwatch gate\n\n"
        "## Docs\n\n"
        "- docs/models.md \u2014 add section\n\n"
        "## Changes\n\n"
        "- authoring_evidence.py\n"
    )
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "name": "b17_clauses",
                      "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md",
                   "All confirmed.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", design_text)
    ctx.write_json("brief.json", {"name": "b17_clauses", "brief": "split constraints"})
    verdict = ops.gate_design_review(ctx)
    assert verdict == "proceed"
    advisory = ctx.facts["design_review_advisory"]
    assert "create_ops.py:1696" in advisory
    assert "authoring_evidence.py:154" not in advisory


def test_a_design_whose_title_is_the_run_name_is_advisory(tmp_path):
    """The gate notes a title that equals the run name, as advisory."""
    import refactory_dsh.design_ops as ops

    design_text = (
        "# b17_clauses\n\n"
        "## Summary\n\nDo the thing.\n\n"
        "## Ask mapping\n\n| Ask (quoted from brief) | Design section | Test |\n|---|---|---|\n"
        "| \"the brief\" | § Changes | test_tool.py |\n\n"
        "## Rule and sites\n\n"
        "- `file.py:1` \u2014 **[changed]**\n\n"
        "## Docs\n\n"
        "- docs/models.md\n\n"
        "## Changes\n\n"
        "- file.py\n"
    )
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "name": "b17_clauses",
                      "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md", "All good.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", design_text)
    ctx.write_json("brief.json", {"name": "b17_clauses", "brief": "split constraints"})
    verdict = ops.gate_design_review(ctx)
    assert verdict == "proceed"
    assert "title is the run name" in ctx.facts["design_review_advisory"]


def test_a_design_with_left_no_reason_is_advisory(tmp_path):
    """The gate catches a rule site marked [left] without a reason."""
    import refactory_dsh.design_ops as ops

    design_text = (
        "# Enforce clause coverage\n\n"
        "## Summary\n\nFix the audit.\n\n"
        "## Ask mapping\n\n| Ask (quoted from brief) | Design section | Test |\n|---|---|---|\n"
        "| \"the brief\" | § Changes | test_tool.py |\n\n"
        "## Rule and sites\n\n"
        "- `evidence.py:10` \u2014 **[changed]**\n"
        "- `ops.py:20` \u2014 **[left]**\n\n"
        "## Docs\n\n"
        "- docs/models.md\n\n"
        "## Changes\n\n"
        "- evidence.py\n"
    )
    ctx = nn.RunContext(tmp_path)
    ctx.facts.update({"source": str(tmp_path), "name": "b17_clauses",
                      "design_review_attempts_used": 0})
    (tmp_path / "reviews").mkdir(parents=True, exist_ok=True)
    ctx.write_text("reviews/design-review.md", "All good.\n\nVERDICT: PASS\n")
    ctx.write_text("design.md", design_text)
    ctx.write_json("brief.json", {"name": "b17_clauses", "brief": "fix audit"})
    verdict = ops.gate_design_review(ctx)
    assert verdict == "proceed"
    assert "[left] without a reason" in ctx.facts["design_review_advisory"]


# ── design_fast, on the same steps (`repo_change`) ─────────────────────────────
#: What design_fast's strategist and focused QA answer; the coder's answers are each test's own.
FAST_PLAN = "Change only the requested file."
FAST_QA = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                    "status": "pass", "evidence": "exit 0"}],
                      "findings": [], "not_run": []})


def _fast(workdir: Path, *emits: str, refused: int = 0, knobs: dict | None = None, **task: str):
    """A design_fast run: the strategist, then each emission and, once one applies, its focused QA.

    The first ``refused`` emissions are ones materialise cannot apply, so no QA follows them.
    """
    mode = DesignFastMode()
    state = mode.state_dict()
    for name, value in (knobs or {}).items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    answers = [FAST_PLAN] + [answer for index, emit in enumerate(emits)
                             for answer in ((emit,) if index < refused
                                            else (emit, FAST_QA, VERIFICATION_ANSWER))]
    return mode.run(workdir, model=FakeModel(answers),
                    task={"name": NAME, "brief": "a tool", "check": "python tool.py", **task})


def test_design_fast_changes_a_repository_in_a_worktree_and_delivers_nothing(repo, tmp_path):
    """It works where design works, and its work stays on the run's branch: it has no gate to approve a PR."""
    (repo / "app.py").write_text("print('mine, not committed')\n")
    head = git(repo, "rev-parse", "HEAD")
    # Passes only where app.py is as committed and the setup command has run: the worktree, prepared.
    tool = ("import pathlib\nassert pathlib.Path('app.py').read_text() == \"print('app')\\n\"\n"
            "assert pathlib.Path('built.txt').exists()\n")
    model = FakeModel([FAST_PLAN, render({"tool.py": tool}), FAST_QA, VERIFICATION_ANSWER])
    result = DesignFastMode().run(tmp_path / "run", model=model, task={
        "name": NAME, "brief": "a tool", "check": "python tool.py", "target": str(repo),
        "setup": "python -c \"open('built.txt', 'w').write('built')\""})
    assert result.exit_ok, result.error
    assert "a clone of the run's worktree at its last attempt" in model.prompts[1], \
        "emit is told it works in a clone of the last attempt"
    assert str(worktree.WRAPPER) in model.prompts[2], "focused QA is told how to run a command in the worktree"
    assert (git(repo, "rev-parse", "HEAD"), (repo / "app.py").read_text()) == (head, "print('mine, not committed')\n")
    payload = _report(tmp_path / "run")
    assert (payload["target"], payload["base"]) == (str(repo.resolve()), head)
    assert git(repo, "ls-tree", "--name-only", payload["branch"]).splitlines() == ["app.py", "tool.py"]
    assert "fast" in payload and payload["source_project"] is None
    assert not (tmp_path / "run" / "delivery.json").exists() and not (tmp_path / "run" / "delivery.md").exists()
    [entry] = projects.load()["projects"]
    assert entry["runs"][-1]["status"] == "succeeded"


def test_design_fast_with_no_target_starts_a_new_project(tmp_path):
    result = _fast(tmp_path / "run", render({"tool.py": "print('built')\n"}))
    assert result.exit_ok, result.error
    project = (projects_dir() / NAME).resolve()
    payload = _report(tmp_path / "run")
    assert payload["target"] == str(project)
    assert git(project, "ls-tree", "--name-only", payload["branch"]) == "tool.py"


def test_design_fast_edits_its_last_attempt_and_reviews_the_whole_branch(tmp_path):
    """A repair is the next commit, made with an edit, and focused QA is shown every file the branch changes."""
    failing = render({"tool.py": "raise SystemExit(1)\n", "helper.py": "x = 1\n"})
    fixed = _edit("tool.py", ("raise SystemExit(1)\n", "print('fixed')\n"))
    result = _fast(tmp_path / "run", failing, fixed)
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    project = Path(payload["target"])
    assert git(project, "rev-list", "--count", f"{payload['base']}..{payload['branch']}") == "2"
    assert git(project, "show", f"{payload['branch']}:tool.py") == "print('fixed')"
    assert json.loads(result.facts["review_files"]) == ["helper.py", "tool.py"]
    assert payload["fast"]["repairs_used"] == 1


def test_design_fast_carries_the_lockfile_its_install_writes(repo, tmp_path, monkeypatch):
    """The check's install amends the attempt, and focused QA is shown that file too."""
    stub(tmp_path / "bin", "npm", "echo '{\"lockfileVersion\": 3}' > package-lock.json")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    result = _fast(tmp_path / "run", render({"package.json": "{}\n", "tool.py": "print('built')\n"}),
                   target=str(repo))
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert "package-lock.json" in git(repo, "ls-tree", "--name-only", payload["branch"]).splitlines()
    assert payload["head"] == git(repo, "rev-parse", payload["branch"]) == result.facts["commit"]
    assert "package-lock.json" in json.loads(result.facts["review_files"])

def test_design_fast_carries_generate_changes(repo, tmp_path):
    """The generate command amends the attempt, and focused QA is shown that file too."""
    _commit(repo, "generated.txt", "old\n")
    result = _fast(tmp_path / "run", render({"tool.py": "print('built')\n"}),
                   target=str(repo), generate="echo new > generated.txt")
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert "generated.txt" in git(repo, "ls-tree", "--name-only", payload["branch"]).splitlines()
    assert payload["head"] == git(repo, "rev-parse", payload["branch"]) == result.facts["commit"]
    assert "generated.txt" in json.loads(result.facts["review_files"])


def test_design_fast_sends_an_emission_it_cannot_apply_back_to_emit(tmp_path):
    """A refused emission does not spend a repair attempt; the gate's iteration limit bounds retries.

    The builder's file holds text copied from a truncated read: the derived emission
    carries it, and materialise refuses it as it refused the block.
    """
    truncated = "===FILE: tool.py===\nx = 1  # ... (line truncated to 2000 chars)\n"
    result = _fast(tmp_path / "run", truncated, render({"tool.py": "print('whole')\n"}), refused=1)
    assert result.exit_ok, result.error
    # The feedback reaches emit through the repair_feedback fact, not through repair.txt.
    assert result.facts.get("repair_feedback") and "cut text short" in result.facts["repair_feedback"]
    # A refused emission does not spend a repair attempt.
    assert _report(tmp_path / "run")["fast"]["repairs_used"] == 0


def test_design_fast_with_its_worktree_off_keeps_its_own_copy_and_makes_no_worktree(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path / "project"))
    result = _fast(tmp_path / "run", "===FILE: tool.py===\nx = 1\n", knobs={"worktree": False})
    assert result.exit_ok, result.error
    assert result.facts["artifact"] == str((tmp_path / "project" / repo_change.ARTIFACTS / NAME).resolve())
    assert not worktree.root().exists() and projects.load()["projects"] == []


def test_design_fast_copies_a_source_project_and_refuses_a_target_with_it(repo, tmp_path, monkeypatch):
    """`source_project` is the other way to start from existing code, so a run takes one of the two."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path / "project"))
    truncated = "===FILE: tool.py===\nx = 1  # ... (line truncated to 2000 chars)\n"
    copied = _fast(tmp_path / "run", truncated, "===FILE: tool.py===\nx = 1\n", refused=1,
                   source_project=str(repo))
    assert copied.exit_ok, copied.error
    assert "cut text short" in str(copied.facts.get("repair_feedback", "")), "a refusal is reported here too"
    assert sorted(path.name for path in Path(copied.facts["artifact"]).iterdir()) == ["app.py", "tool.py"]
    assert not worktree.root().exists()
    refused = _fast(tmp_path / "both", source_project=str(repo), target=str(repo))
    assert not refused.exit_ok
    assert "source_project copies a directory into the artifact, and target" in str(refused.error)


def test_design_fast_refuses_to_follow_up_a_pull_request(tmp_path):
    """Following one up updates it, and design_fast delivers nothing, so it is refused, not ignored."""
    result = _fast(tmp_path / "run", refine="acme/app#3")
    assert not result.exit_ok and "design_fast delivers nothing" in str(result.error)


def test_the_direct_entry_point_passes_the_worktree_flags_to_design_fast(monkeypatch, tmp_path, capsys):
    from refactory_dsh import design

    seen: dict = {}

    def run(self, workdir, **kwargs):
        seen.update(knobs=self.values(), task=kwargs["task"])
        return nn.RunResult(exit_ok=True, facts={})

    monkeypatch.setattr(DesignFastMode, "run", run)
    # The walker profile check needs a headless profile; create one under a
    # fake DSH home so the test does not depend on the real ~/.dsh directory.
    _fake_dsh = tmp_path / "fake_dsh"
    (_fake_dsh / "profiles" / "headless").mkdir(parents=True)
    (_fake_dsh / "profiles" / "headless" / "package.json").write_text("{}")
    monkeypatch.setattr("refactory_dsh.dsh_profile.model_home", lambda home=None: _fake_dsh)
    monkeypatch.chdir(tmp_path)
    assert design.main(["--name", NAME, "--brief", "b", "--check", "true", "--fast", "--no-worktree"]) == 0
    assert seen["knobs"]["worktree"] is False
    with pytest.raises(SystemExit) as refused:
        design.main(["--name", NAME, "--brief", "b", "--check", "true", "--fast", "--no-pr"])
    assert refused.value.code == 2 and "--fast does not take --no-pr" in capsys.readouterr().err


def test_ceo_passes_the_worktree_flags_to_design_and_refuses_them_where_they_cannot_apply(
        monkeypatch, tmp_path, capsys):
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_forward", lambda module, argv: seen.append(argv) or 0)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["ceo", "a tool", "--check", "true", "--target", ".", "--base", "v1", "--setup", "make",
                     "--generate", "make"]) == 0
    assert {f"target={tmp_path.resolve()}", "base=v1", "setup=make", "generate=make"} <= set(seen[0])
    assert cli.main(["ceo", "a tool", "--check", "true", "--no-worktree"]) == 0
    assert "worktree=0" in seen[1]
    assert cli.main(["ceo", "a tool", "--check", "true", "--mode", "design_fast", "--target", ".",
                     "--setup", "make", "--generate", "make"]) == 0
    assert {f"target={tmp_path.resolve()}", "setup=make", "generate=make"} <= set(seen[2]), "design_fast takes them too"
    for refused in (["--mode", "create", "--single", "--target", "."], ["--no-worktree", "--target", "."],
                    ["--mode", "design_fast", "--source-project", ".", "--target", "."],
                    ["--mode", "design_fast", "--source-project", ".", "--generate", "make"]):
        assert cli.main(["ceo", "a tool", "--check", "true", *refused]) == 2
    err = capsys.readouterr().err
    assert "does not take --target" in err and "--no-worktree cannot take --target" in err
    assert "--source-project cannot take --target" in err
    assert "--source-project cannot take --generate" in err
    assert len(seen) == 3, "a refused flag forwards nothing"
    assert cli.main(["ceo", "default_pack/design", "--target", "."]) == 0
    assert f"target={tmp_path.resolve()}" in seen[3], "a named mode takes them as knobs"


def test_the_console_design_verb_takes_a_target_too(monkeypatch, tmp_path):
    """The GUI chat's headless route and the refactory tool's `design` action go through it."""
    seen: list[list[str]] = []
    monkeypatch.setattr(console, "_forward", lambda module, argv: seen.append(argv) or {})
    monkeypatch.chdir(tmp_path)
    console.main(["design", "--name", "x", "--brief", "b", "--check", "true", "--target", ".",
                  "--base", "v1", "--setup", "make", "--generate", "make"])
    assert seen[0][seen[0].index("--target") + 1] == str(tmp_path.resolve())
    assert {"--base", "v1", "--setup", "make", "--generate", "make"} <= set(seen[0])


# ── base-comparison gate for inherited test failures (dm-80, dm-82) ─────────


def test_inherited_failure_passes_check(tmp_path):
    """A base with one failing test and an attempt that fixes what the brief asks
    passes the check, with the inherited failure named."""
    repo = git_repo(tmp_path / "repo")
    _commit(repo, "test_base_broken.py", "def test_base_broken():\n    assert False\n")
    result = _design(
        tmp_path / "run",
        FakeModel(_answers(render({"tool.py": "print('built')\n"}))),
        target=str(repo),
        check="python -m pytest -q",
    )
    assert result.exit_ok, result.error
    check_txt = (tmp_path / "run" / "check.txt").read_text()
    assert "inherited" in check_txt
    assert "test_base_broken" in check_txt


def test_inherited_failure_does_not_excuse_a_check_that_stopped_at_it(tmp_path):
    """With -x nothing after the inherited failure ran, so the attempt is not excused."""
    repo = git_repo(tmp_path / "repo")
    _commit(repo, "test_base_broken.py", "def test_base_broken():\n    assert False\n")
    fixed = render({"tool.py": "print('built')\n",
                    "test_base_broken.py": "def test_base_broken():\n    import tool\n    assert hasattr(tool, '__file__')\n"})
    result = _design(
        tmp_path / "run",
        FakeModel(_answers(render({"tool.py": "print('built')\n"}), fixed)),
        target=str(repo),
        check="python -m pytest -q -x",
    )
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert git(Path(payload["target"]), "rev-list", "--count",
               f"{payload['base']}..{payload['branch']}") == "2"


def _self_checks(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "self-check-record.jsonl").read_text().splitlines()]


def test_new_failure_introduced_by_attempt_is_refused(tmp_path):
    """An attempt that adds a new failure goes back to the builder, and the base's own
    failure does not. Since b41a the self-check sends it back before the reviews, as a
    fix-up amended into the attempt, so the branch keeps one commit."""
    repo = git_repo(tmp_path / "repo")
    _commit(repo, "test_base_broken.py", "def test_base_broken():\n    assert False\n")
    failing = render({"tool.py": "print('built')\n",
                      "test_new.py": "def test_new_broken():\n    assert False\n"})
    fixed = render({"tool.py": "print('built')\n",
                    "test_new.py": "def test_new_broken():\n    import tool\n    assert hasattr(tool, '__file__')\n"})
    result = _design(
        tmp_path / "run",
        FakeModel(_answers(failing, fixed)),
        target=str(repo),
        check="python -m pytest -q",
    )
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    project = Path(payload["target"])
    assert git(project, "rev-list", "--count",
               f"{payload['base']}..{payload['branch']}") == "1"
    sent_back = _self_checks(tmp_path / "run")[0]
    assert sent_back["outcome"] == "fixup"
    assert sent_back["failing_ids"] == ["test_new.py::test_new_broken"]
    assert sent_back["inherited_filtered"] == ["test_base_broken.py::test_base_broken"]


def test_brief_mentioned_failure_not_excused(tmp_path):
    """A run whose brief is to fix the failing test is not excused: the base fails it too,
    but the self-check sends it back (b41a)."""
    repo = git_repo(tmp_path / "repo")
    _commit(repo, "test_fix_me.py", "def test_fix_me():\n    assert False\n")
    not_fixed = render({"tool.py": "print('did not fix it')\n"})
    fixed = render({"test_fix_me.py": "def test_fix_me():\n    import tool\n    assert hasattr(tool, '__file__')\n"})
    result = DesignMode().run(
        tmp_path / "run",
        model=FakeModel(_answers(not_fixed, fixed)),
        verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "fix test_fix_me",
              "check": "python -m pytest -q", "target": str(repo)},
    )
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    project = Path(payload["target"])
    assert git(project, "rev-list", "--count",
               f"{payload['base']}..{payload['branch']}") == "1"
    sent_back = _self_checks(tmp_path / "run")[0]
    assert sent_back["failing_ids"] == ["test_fix_me.py::test_fix_me"]
    assert sent_back["outcome"] == "fixup" and sent_back["inherited_filtered"] == []


def test_failure_in_a_test_file_the_attempt_changes_is_not_excused():
    """A test file the attempt edits is the attempt's, wherever the runner ran from."""
    from refactory_dsh.base_check import compare_failures
    failing = {"tests/test_a.py::test_x"}
    edited = compare_failures(failing, failing, ["pkg/tests/test_a.py"], "", "")
    assert edited.new_failures == failing and not edited.all_excused
    untouched = compare_failures(failing, failing, ["pkg/tool.py"], "", "")
    assert untouched.inherited == failing and untouched.all_excused


def test_inherited_failures_excuse_a_check_only_when_nothing_else_failed_or_went_unrun():
    """Matched names are not enough: the last step must be the runner, run to completion."""
    from refactory_dsh.base_check import excusal_blocker, parse_test_failures
    out = "FAILED tests/test_a.py::test_x - assert False\n1 failed, 3 passed in 0.12s\n"
    failures = parse_test_failures(out)
    assert excusal_blocker("pnpm test && .venv/bin/python -m pytest -q tests 2>&1", out, failures) is None
    assert excusal_blocker("python -m pytest -q -rfEx", out, failures) is None
    # A step after the runner, or one the runner's failure skipped, is not accounted for.
    assert "last step" in excusal_blocker("python -m pytest -q; ruff check .", out, failures)
    vitest = " FAIL  tests/a.spec.ts > suite > works\n      Tests  1 failed | 9 passed (10)\n"
    assert "never ran" in excusal_blocker("npx vitest run && pytest -q", vitest, parse_test_failures(vitest))
    assert excusal_blocker("npx vitest run", vitest, parse_test_failures(vitest)) is None
    # Stopping at the first failure hides every test after it.
    assert "first failure" in excusal_blocker("python -m pytest -xq", out, failures)
    # An error the summary counts but no name accounts for, or no summary at all.
    errored = out.replace("1 failed, 3 passed", "1 failed, 3 passed, 1 error")
    assert "not identified" in excusal_blocker("pytest -q", errored, failures)
    assert "not identified" in excusal_blocker("pytest -q", "FAILED tests/test_a.py::test_x - x\n", failures)


def test_a_test_module_the_attempt_breaks_is_its_failure():
    """A collection error is a failure with a name; the base does not have it."""
    from refactory_dsh.base_check import compare_failures, parse_test_failures
    base = parse_test_failures("FAILED tests/test_a.py::test_x - x\n1 failed in 0.1s\n")
    attempt = parse_test_failures("FAILED tests/test_a.py::test_x - x\n"
                                  "ERROR tests/test_b.py - ImportError: cannot import name\n"
                                  "1 failed, 1 error in 0.1s\n")
    assert attempt == {"tests/test_a.py::test_x", "tests/test_b.py"}
    result = compare_failures(base, attempt, [], "", "")
    assert result.new_failures == {"tests/test_b.py"} and not result.all_excused


def test_the_base_is_prepared_with_its_own_home_not_the_attempts(tmp_path):
    """A setup that writes into HOME and keeps what it finds there (as a tool's profile link
    does) must not leave the attempt's checks reading the base's.

    The first attempt fails in generate, before its own setup runs, so the base comparison
    prepares the base first. Sharing a HOME, the base's marker would stay, and the second
    attempt's check would fail on it however good the emission (b20b, dm-88)."""
    repo = git_repo(tmp_path / "repo")
    broken = render({"tool.py": "print('built')\n", "gen.sh": "exit 1\n"})
    fixed = render({"tool.py": "print('built')\n", "gen.sh": "exit 0\n"})
    result = _design(
        tmp_path / "run",
        FakeModel(_answers(broken, fixed)),
        _mode(generate="[ ! -f gen.sh ] || sh gen.sh",
              setup='[ -f "$HOME/built-in" ] || pwd > "$HOME/built-in"'),
        target=str(repo),
        check='test "$(cat "$HOME/built-in")" = "$(pwd)"',
    )
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    assert git(Path(payload["target"]), "rev-list", "--count",
               f"{payload['base']}..{payload['branch']}") == "2"
    attempt_home = tmp_path / "run" / ".refactory" / "check-home" / "built-in"
    base_home = tmp_path / "run" / ".refactory" / "base-check-home" / "built-in"
    assert attempt_home.read_text().strip() != base_home.read_text().strip()


def test_base_check_cannot_run_stops_unavailable(tmp_path):
    """A deterministic base-preparation failure stops the run as unavailable.

    The setup command ``sh setup.sh`` references a file that only exists in
    the attempt's emission — on the base commit setup.sh is absent, so the
    base preparation fails with a deterministic exit code (not a signal crash).
    Per v5 correction 2, deterministic failures stop as unavailable; only
    signal crashes continue without excusals (tested in test_robustness.py).
    """
    repo = git_repo(tmp_path / "repo")
    result = DesignMode().run(
        tmp_path / "run",
        model=FakeModel(_answers(render({
            "tool.py": "raise SystemExit(1)\n",
            "setup.sh": "#!/bin/sh\necho ok\n",
        }))),
        verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "a tool",
              "check": "python tool.py", "target": str(repo),
              "setup": "sh setup.sh"},
    )
    assert result.exit_ok is False
    error = result.error or ""
    assert ("unavailable" in error.lower() or "base" in error.lower()
            or "setup" in error.lower())
    assert not result.facts.get("attempts_used")


def test_design_named_failure_not_excused(tmp_path):
    """A test named only in design.md, failing on the base too, is the base's: design mode's
    one rule (b48a) lets the brief and the corrections keep a test, not the design. The
    self-check calls it inherited and the attempt goes on; a brief that names it keeps it
    (`test_a_brief_named_failure_is_not_excused`)."""
    repo = git_repo(tmp_path / "repo")
    _commit(repo, "test_oblique.py", "def test_oblique():\n    assert False\n")
    not_fixed = render({"tool.py": "print('did not fix it')\n"})
    fixed = render({"test_oblique.py": "def test_oblique():\n    import tool\n    assert hasattr(tool, '__file__')\n"})

    naming_appendix = "\n\n## Tests to fix\n- test_oblique.py::test_oblique\n"

    class DesignNamingModel(FakeModel):
        """Intercepts the design_doc prompt to write a design.md that names the test."""
        def complete(self, prompt, *, system="", workdir=None, **workspace):
            answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
            if "Write the design document" in prompt and workdir is not None:
                design_path = Path(workdir) / "design.md"
                if design_path.exists():
                    existing = design_path.read_text()
                else:
                    existing = answer
                design_path.write_text(existing + naming_appendix)
            return answer

    result = DesignMode().run(
        tmp_path / "run",
        model=DesignNamingModel(_answers(not_fixed, fixed,
                                         )),
        verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "the validation test for contract X fails",
              "check": "python -m pytest -q", "target": str(repo)},
    )
    assert result.exit_ok, result.error
    assert "test_oblique.py::test_oblique" in (tmp_path / "run" / "design.md").read_text()
    [inherited] = _self_checks(tmp_path / "run")
    assert inherited["outcome"] == "inherited" and inherited["failing_ids"] == []
    assert inherited["inherited_filtered"] == ["test_oblique.py::test_oblique"]
    assert result.facts["inherited_failures"] == ["test_oblique.py::test_oblique"]


def test_a_brief_named_failure_is_not_excused(tmp_path):
    """A test the brief names is not excused even when the base fails it too: the self-check
    sends it back (b41a, b48a)."""
    repo = git_repo(tmp_path / "repo")
    _commit(repo, "test_oblique.py", "def test_oblique():\n    assert False\n")
    not_fixed = render({"tool.py": "print('did not fix it')\n"})
    fixed = render({"test_oblique.py": "def test_oblique():\n    import tool\n    assert hasattr(tool, '__file__')\n"})
    result = DesignMode().run(
        tmp_path / "run",
        model=FakeModel(_answers(not_fixed, fixed)),
        verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "fix test_oblique.py::test_oblique — the validation test for contract X fails",
              "check": "python -m pytest -q", "target": str(repo)},
    )
    assert result.exit_ok, result.error
    payload = _report(tmp_path / "run")
    project = Path(payload["target"])
    assert git(project, "rev-list", "--count",
               f"{payload['base']}..{payload['branch']}") == "1"
    sent_back = _self_checks(tmp_path / "run")[0]
    assert sent_back["failing_ids"] == ["test_oblique.py::test_oblique"]
    assert sent_back["outcome"] == "fixup" and sent_back["inherited_filtered"] == []


def test_guardedness_record_shows_new_test_failing_on_base(tmp_path):
    """dm-82: the record shows a new test failing on the base, and passing with the attempt."""
    repo = git_repo(tmp_path / "repo")
    result = _design(
        tmp_path / "run",
        FakeModel(_answers(render({
            "tool.py": "BUILT = True\n",
            "test_tool.py": "import tool\ndef test_tool_built():\n    assert tool.BUILT\n",
        }))),
        target=str(repo),
        check="python -m pytest -q",
    )
    assert result.exit_ok, result.error
    guardedness = json.loads((tmp_path / "run" / "test-guardedness.json").read_text())
    assert "test_tool.py" in guardedness["new_test_files"]
    found_guarding = False
    for file_record in guardedness["results"]:
        if file_record["file"] == "test_tool.py":
            if file_record.get("tests"):
                for t in file_record["tests"]:
                    if t.get("on_base") == "fail":
                        found_guarding = True
            elif file_record.get("guarding"):
                found_guarding = True
    # The test imports `tool` which does not exist on the base, so it either
    # fails on import (cannot_run_on_base) or fails on assertion — either way
    # the record exists and the file is listed.
    assert found_guarding or any(
        r["file"] == "test_tool.py" and r.get("file_level") == "cannot_run_on_base"
        for r in guardedness["results"]
    ), f"test_tool.py should be guarding or cannot_run_on_base but got: {guardedness}"


# ── T1: @-paths accepted in all three emission block kinds ─────────────────
def test_at_paths_are_accepted_in_emission():
    at_path = "patches/@scope@1.0.0.patch"
    # ===FILE: with body
    files, _edits, _deletes = repo_change._emitted(
        f"===FILE: {at_path}===\npatch content\n", edits=True)
    assert Path(at_path) in files

    # ===EDIT: with SEARCH/DIVIDER/REPLACE
    edit_body = (
        f"===EDIT: {at_path}===\n"
        f"{repo_change.SEARCH}\nold line\n{repo_change.DIVIDER}\nnew line\n{repo_change.REPLACE}\n"
    )
    _files2, edits2, _deletes2 = repo_change._emitted(edit_body, edits=True)
    assert Path(at_path) in edits2

    # ===DELETE:
    _files3, _edits3, deletes3 = repo_change._emitted(
        f"===DELETE: {at_path}===\n", edits=True)
    assert Path(at_path) in deletes3


# ── T3: ..-path refused, naming the path ───────────────────────────────────
def test_dotdot_path_is_refused_naming_the_path():
    import re as _re
    with pytest.raises(worktree.AttemptRefused, match=_re.escape("../etc/passwd")):
        repo_change._emitted("===FILE: ../etc/passwd===\ncontent\n", edits=True)


# ── T4: absolute path refused, naming the path ────────────────────────────
def test_absolute_path_is_refused_naming_the_path():
    import re as _re
    with pytest.raises(worktree.AttemptRefused, match=_re.escape("/etc/passwd")):
        repo_change._emitted("===FILE: /etc/passwd===\ncontent\n", edits=True)


# ── dm-208: a control or format character in a path is refused by name ─────
#: One character of each class a path may not hold, and how a refusal names it.
UNPRINTABLE = [("\x7f", "U+007F (a control character)"), ("\x9b", "U+009B (a control character)"),
               ("\u202e", "U+202E RIGHT-TO-LEFT OVERRIDE"), ("\u200f", "U+200F RIGHT-TO-LEFT MARK"),
               ("\u2067", "U+2067 RIGHT-TO-LEFT ISOLATE"), ("\u061c", "U+061C ARABIC LETTER MARK"),
               ("\u200b", "U+200B ZERO WIDTH SPACE"), ("\u2060", "U+2060 WORD JOINER"),
               ("\ufeff", "U+FEFF ZERO WIDTH NO-BREAK SPACE")]
UNPRINTABLE_IDS = ["DEL", "C1-control", "bidi-override", "bidi-mark", "bidi-isolate", "arabic-mark",
                   "zero-width-space", "word-joiner", "byte-order-mark"]


@pytest.mark.parametrize("char, named", UNPRINTABLE, ids=UNPRINTABLE_IDS)
def test_a_path_holding_a_control_or_format_character_is_refused_by_name(char, named):
    import re as _re
    with pytest.raises(worktree.AttemptRefused, match=_re.escape(named)):
        repo_change._emitted(f"===FILE: src/a{char}b.py===\ncontent\n", edits=True)
    with pytest.raises(worktree.AttemptRefused, match=_re.escape(named)):
        repo_change._emitted(f"===DELETE: src/a{char}b.py===\n", edits=True)


def test_a_path_with_a_joiner_that_names_are_written_with_is_accepted():
    """U+200C joins letters in Persian names; it is a format character a path may hold."""
    persian = "docs/\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645.md"
    files, _edits, _deletes = repo_change._emitted(f"===FILE: {persian}===\ncontent\n", edits=True)
    assert Path(persian) in files
    assert repo_change.unprintable_refusal(persian) == ""


# ── T5: malformed header refused, quoting the line in _emitted ─────────────
@pytest.mark.parametrize("header, fragment", [
    ("===FILE: has space.txt===", "has space.txt"),
    ("===FILE:===", "===FILE:==="),
])
def test_malformed_header_is_refused_quoting_the_line_in_emitted(header, fragment):
    import re as _re
    with pytest.raises(worktree.AttemptRefused, match=_re.escape(fragment)):
        repo_change._emitted(f"{header}\ncontent\n", edits=True)


# ── T9: backtick- and quote-wrapped paths refused in _emitted ──────────────
@pytest.mark.parametrize("header, fragment", [
    ("===FILE: `src/a.py`===", "`src/a.py`"),
    ("===FILE: 'src/a.py'===", "'src/a.py'"),
    ('===FILE: "src/a.py"===', '"src/a.py"'),
])
def test_backtick_and_quote_wrapped_paths_are_refused_in_emitted(header, fragment):
    import re as _re
    with pytest.raises(worktree.AttemptRefused, match=_re.escape(fragment)):
        repo_change._emitted(f"{header}\ncontent\n", edits=True)


# ── T10: worktree apply path with @-path files (all three block kinds) ─────
def test_worktree_apply_at_path_all_three_block_kinds(tmp_path):
    """The case b36b needed: FILE, EDIT and DELETE of @-path files through the
    worktree apply path, committed in the attempt, and the committed tree
    asserted.  Uses the real scoped-package name shape."""
    repo = git_repo(tmp_path / "repo")
    patch_dir = Path(repo) / "patches"
    patch_dir.mkdir()
    existing = "@deepseek-ai__dsh-subagent-claude-code@0.2.0-rc.2.patch"
    to_delete = "@scope__other@2.0.0.patch"
    (patch_dir / existing).write_text("old patch\n")
    (patch_dir / to_delete).write_text("delete me\n")
    git(repo, "add", "patches")
    git(repo, "commit", "-q", "-m", "add patches")

    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    found = worktree.create(repo, "HEAD", "atat0000aaaa", run)

    new_file = "patches/@scope__new@3.0.0.patch"
    edit_file = f"patches/{existing}"
    delete_file = f"patches/{to_delete}"
    emission = (
        f"===FILE: {new_file}===\nnew patch content\n"
        f"===EDIT: {edit_file}===\n"
        f"{repo_change.SEARCH}\nold patch\n"
        f"{repo_change.DIVIDER}\nnew patch\n"
        f"{repo_change.REPLACE}\n"
        f"===DELETE: {delete_file}===\n"
    )

    files, deletes, _rev, _repeated = repo_change.apply_emission(emission, found, run)
    advanced = worktree.commit_attempt(run, files, "attempt 1", deletes)
    wt = Path(found.path)

    # ===FILE: new @-path file is committed
    assert git(wt, "show", f"{advanced.head}:{new_file}") == "new patch content"
    # ===EDIT: existing @-path file is updated
    assert git(wt, "show", f"{advanced.head}:{edit_file}") == "new patch"
    # ===DELETE: @-path file is removed from the committed tree
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        git(wt, "show", f"{advanced.head}:{delete_file}")


def test_the_design_mode_declares_the_knobs_the_ceo_passes():
    """`--target` reaches a knob the graph consumes, not one the runtime would refuse."""
    state = registry.resolve(REPO_ROOT, "default_pack/design")[1]().state_dict()
    state = interactive.with_knobs(state, ["target=/somewhere", "base=v1", "worktree=0", "setup=make",
                                           "generate=make"])
    assert (state["knobs"]["target"]["value"], state["knobs"]["base"]["value"],
            state["knobs"]["worktree"]["value"], state["knobs"]["setup"]["value"],
            state["knobs"]["generate"]["value"]) == (
        "/somewhere", "v1", False, "make", "make")
