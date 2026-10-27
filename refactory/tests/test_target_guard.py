"""Design runs watch the target they build for (#330 PR 4a).

Three failures from shared checkouts: a run's baseline moved under it with nobody told
(I1), two runs with one artifact name overwrote each other's artifact (K4), and the
`emit` worker edited the repository directly instead of answering through `emit.txt`,
so its changes skipped materialise, the check and the reviews (#319).

These runs have their ``worktree`` knob off: that is the path where this guard is the
whole protection. A run with its own worktree (#351) keeps it as a backstop, and
`test_design_worktree.py` covers that.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import agent_delivery, design_ops, nn, repo_change, target_guard
from refactory_dsh.design import DesignMode
from tests_support import DESIGN_DOC_ANSWER, PITFALLS_ANSWER, RESEARCH_ANSWER, VERIFICATION_ANSWER, REVIEW_ANSWERS, FakeModel, StaleEmitModel, render

NAME = "guarded_tool"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def target(tmp_path, monkeypatch) -> Path:
    """A git checkout on `main` with one commit, as the project nodes resolve to."""
    root = tmp_path / "target"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    (root / "app.py").write_text("print('app')\n")
    _git(root, "add", "app.py")
    _git(root, "commit", "-q", "-m", "base")
    monkeypatch.setenv("REFACTORY_PROJECT", str(root))
    return root


def _prepare(workdir: Path, name: str = NAME) -> nn.RunContext:
    """Prepare a design run in ``workdir``; a stand-in model skips the readiness probe."""
    ctx = nn.RunContext(workdir, model=FakeModel([]), knobs={"worktree": False},
                        task={"name": name, "brief": "a tool", "check": "python tool.py"})
    ctx.facts.update(design_ops.prepare_design(ctx))
    return ctx


def _without_worktree() -> DesignMode:
    """Design mode with its ``worktree`` knob off: the artifact directory, as before #351."""
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["worktree"]["value"] = False
    mode.load_state_dict(state)
    return mode


def _spine(workdir: Path, status: str, owner: int | None = None) -> None:
    """Leave the status the spine writes for a run it drives, and its owner process."""
    (workdir / ".refactory").mkdir(parents=True, exist_ok=True)
    (workdir / ".refactory" / "spine-status.json").write_text(json.dumps({"status": status, "at": "now"}))
    if owner is not None:
        (workdir / ".refactory" / "spine-checkpoint.json").write_text(json.dumps({"owner_pid": owner}))


def _exited_pid() -> int:
    """The id of a process that has already exited."""
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


# ── I1: the baseline is recorded, and a move is reported ───────────────────
def test_prepare_records_the_targets_commit_branch_and_uncommitted_changes(target, tmp_path):
    (target / "app.py").write_text("print('edited')\n")
    _prepare(tmp_path / "run")
    recorded = json.loads((tmp_path / "run" / target_guard.RECORD).read_text())
    assert recorded["root"] == str(target.resolve())
    assert recorded["head"] == _git(target, "rev-parse", "HEAD")
    assert recorded["branch"] == "main"
    assert list(recorded["changes"]) == ["app.py"], "the dirty state is part of the baseline"
    assert recorded["output_only"] == ["emit", *design_ops.PLANNING_STEPS]


def test_a_later_step_warns_when_the_target_moved_since_prepare(target, tmp_path, capsys):
    ctx = _prepare(tmp_path / "run")
    ctx.write_text("emit.txt", render({"tool.py": "print('built')\n"}))
    repo_change.materialise(ctx)
    assert "warning" not in capsys.readouterr().err, "an unchanged target is not worth a warning"

    _git(target, "checkout", "-q", "-b", "other")
    (target / "notes.txt").write_text("another session's work\n")
    _git(target, "add", "notes.txt")
    _git(target, "commit", "-q", "-m", "elsewhere")
    (target / "app.py").write_text("print('dirty')\n")
    repo_change.materialise(ctx)
    warning = capsys.readouterr().err
    assert warning.startswith("refactory: run warning: the target"), (
        "the marker is what sends it to the CEO rather than only to the log"
    )
    assert "HEAD moved" in warning
    assert "the branch changed from main to other" in warning
    assert "app.py" in warning


def test_the_routine_readiness_notice_stays_a_log_warning(target, tmp_path, capsys, monkeypatch):
    """Only the guard's own warnings carry the marker the spine passes to the CEO."""
    from refactory_dsh import sandbox

    notice = "credential readiness is unverified"
    monkeypatch.setattr(sandbox, "model_readiness", lambda *args, **kwargs: {"warning": notice})
    ctx = nn.RunContext(tmp_path / "run", knobs={"worktree": False},
                        task={"name": NAME, "brief": "a tool", "check": "true"})
    design_ops.prepare_design(ctx)
    err = capsys.readouterr().err
    assert f"refactory: warning: {notice}" in err
    assert "run warning" not in err


def test_a_change_listed_after_a_work_tree_rename_is_still_seen(target, tmp_path):
    """git lists a rename's old path as a field of its own, in either status column."""
    (target / "README.md").write_text("readme\n")
    _git(target, "add", "README.md")
    _git(target, "commit", "-q", "-m", "readme")
    (target / "README.md").rename(target / "READ.md")
    _git(target, "add", "-N", "READ.md")
    (target / "app.py").write_text("print('edited')\n")
    changes = target_guard.state(target)["changes"]
    assert sorted(changes) == ["READ.md", "app.py"]


def test_more_uncommitted_paths_than_the_guard_compares_is_reported_not_ignored(target, tmp_path, monkeypatch):
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"])
    monkeypatch.setattr(target_guard, "LIMIT", 3)
    for index in range(5):
        (target / f"generated_{index}.py").write_text("x = 1\n")
    (workdir / "emit.txt").write_text(render({"tool.py": "print('built')\n"}))
    with pytest.raises(ValueError, match="more than 3 paths now differ"):
        agent_delivery.finish(workdir, "emit")
    assert "more than 3 paths now differ" in (target_guard.moved(workdir) or "")


def test_the_runs_own_writes_and_the_artifact_never_count_as_the_target_moving(target, capsys):
    """A run directory inside the target, outside `.factory`, is still the run's own."""
    ctx = _prepare(target / "runs" / "one")
    ctx.write_text("emit.txt", render({"tool.py": "print('built')\n"}))
    repo_change.materialise(ctx)
    assert (target / repo_change.ARTIFACTS / NAME / "tool.py").exists()
    assert target_guard.moved(ctx.workdir) is None
    assert "warning" not in capsys.readouterr().err


# ── K4: one open run per artifact name ─────────────────────────────────────
def test_a_second_open_run_with_the_same_artifact_name_is_refused(target, tmp_path):
    first = tmp_path / "first"
    _prepare(first)
    _spine(first, "waiting_user")
    with pytest.raises(ValueError, match="belongs to another open run") as refused:
        _prepare(tmp_path / "second")
    assert str(first.resolve()) in str(refused.value), "the refusal names the run that holds the name"
    assert "artifact-owners" in str(refused.value), "and how to release a name an abandoned run holds"
    _prepare(tmp_path / "third", name="another_tool")


def test_a_run_whose_driving_process_exited_no_longer_holds_its_name(target, tmp_path):
    held, released = tmp_path / "held", tmp_path / "released"
    _prepare(held)
    _spine(held, "ready", owner=os.getpid())
    with pytest.raises(ValueError, match="belongs to another open run"):
        _prepare(tmp_path / "second")
    _spine(held, "ready", owner=_exited_pid())
    _prepare(released)


def test_a_walked_run_whose_process_exited_no_longer_holds_its_name(target, tmp_path):
    """Ctrl-C ends a walk without recording an error, so the claiming process decides."""
    walked = tmp_path / "walked"
    _prepare(walked)
    (walked / ".refactory" / "progress.json").write_text(json.dumps({"node": "emit", "error": None}))
    with pytest.raises(ValueError, match="walking"):
        _prepare(tmp_path / "second")
    owners = target / repo_change.OWNERS / f"{NAME}.json"
    claim = json.loads(owners.read_text())
    owners.write_text(json.dumps({**claim, "pid": _exited_pid()}))
    _prepare(tmp_path / "third")


def test_a_resumed_run_cannot_check_an_artifact_a_later_run_replaced(target, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    old = _prepare(first)
    old.write_text("emit.txt", render({"tool.py": "print('first')\n"}))
    old.facts.update(repo_change.materialise(old))
    _spine(first, "failed")
    new = _prepare(second)
    new.write_text("emit.txt", render({"tool.py": "print('second')\n"}))
    repo_change.materialise(new)
    (second / "design.json").write_text("{}\n")

    with pytest.raises(ValueError, match="was claimed by another run"):
        design_ops.check_artifact(old)


def test_a_finished_run_releases_its_name_and_a_resumed_one_cannot_take_it_back(target, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    old = _prepare(first)
    _spine(first, "failed")
    _prepare(second)
    _spine(second, "ready")

    _spine(first, "executing")
    old.write_text("emit.txt", render({"tool.py": "print('stale')\n"}))
    with pytest.raises(ValueError, match="belongs to another open run"):
        repo_change.materialise(old)
    assert not (target / repo_change.ARTIFACTS / NAME).exists(), "nothing was overwritten"


# ── #319: emit answers through emit.txt, or it is refused ──────────────────
@pytest.mark.parametrize("git", [True, False], ids=["git", "plain"])
def test_an_output_only_node_that_changes_the_target_is_refused(tmp_path, git):
    root = tmp_path / "target"
    root.mkdir()
    if git:
        _git(root, "init", "-q")
    (root / "model.py").write_text("ORIGINAL = True\n")
    workdir = tmp_path / "run"
    target_guard.record(workdir, root, output_only=("emit",))

    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"])
    (root / "model.py").write_text("EDITED = True\n")
    (workdir / "emit.txt").write_text("All changes are already applied.\n")
    with pytest.raises(ValueError, match=r"changed while agent 'emit' ran, outside its declared outputs \(emit.txt\): .*model.py"):
        agent_delivery.finish(workdir, "emit")
    receipt = agent_delivery.receipts_in(workdir)["emit"]
    assert receipt["status"] == "failed" and "model.py" in " ".join(receipt["target_changes"])


def test_an_output_only_node_that_commits_its_edits_is_still_refused(target, tmp_path):
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"])
    (target / "app.py").write_text("print('edited and committed')\n")
    _git(target, "commit", "-q", "-am", "the worker's own commit")
    (workdir / "emit.txt").write_text("Done.\n")
    with pytest.raises(ValueError, match="the committed files under it changed"):
        agent_delivery.finish(workdir, "emit")


def test_a_new_branch_at_the_same_commit_does_not_fail_emit_but_is_reported(target, tmp_path):
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"])
    _git(target, "checkout", "-q", "-b", "elsewhere")
    (workdir / "emit.txt").write_text(render({"tool.py": "print('built')\n"}))
    assert agent_delivery.finish(workdir, "emit")["status"] == "delivered"
    assert "the branch changed from main to elsewhere" in (target_guard.moved(workdir) or "")


def test_a_refused_attempt_is_retried_from_a_fresh_snapshot_on_the_ceo_route(target, tmp_path):
    """The CEO's bridge begins with reuse, so a refused attempt's record is still pending."""
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (target / "app.py").write_text("print('kept by the person')\n")
    (workdir / "emit.txt").write_text("Done.\n")
    with pytest.raises(ValueError, match="changed while agent 'emit' ran"):
        agent_delivery.finish(workdir, "emit")

    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (workdir / "emit.txt").write_text(render({"tool.py": "print('built')\n"}))
    assert agent_delivery.finish(workdir, "emit")["status"] == "delivered"


def test_a_refused_attempts_output_is_set_aside_so_the_redo_must_write_its_own(target, tmp_path):
    """On the CEO's route the retry is judged against the snapshot from before the refused
    attempt, so a refused emit.txt left in place counted as the retry's delivery."""
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (target / "app.py").write_text("print('edited in place')\n")
    (workdir / "emit.txt").write_text("All changes are already applied.\n")
    with pytest.raises(ValueError, match="changed while agent 'emit' ran") as refused:
        agent_delivery.finish(workdir, "emit")
    assert not (workdir / "emit.txt").exists(), "the redo's worker must not find the refused answer"
    receipt = agent_delivery.receipts_in(workdir)["emit"]
    assert (workdir / receipt["set_aside"] / "emit.txt").read_text() == "All changes are already applied.\n"
    assert receipt["set_aside"] in str(refused.value), "the refusal says where it went"

    _git(target, "checkout", "--", "app.py")
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    with pytest.raises(ValueError, match="did not deliver required outputs: emit.txt"):
        agent_delivery.finish(workdir, "emit")
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (workdir / "emit.txt").write_text(render({"tool.py": "print('built')\n"}))
    assert agent_delivery.finish(workdir, "emit")["status"] == "delivered"


def test_a_target_its_enclosing_checkout_ignores_is_still_watched(tmp_path):
    outer = tmp_path / "outer"
    root = outer / "ignored" / "target"
    root.mkdir(parents=True)
    _git(outer, "init", "-q")
    (outer / ".gitignore").write_text("ignored/\n")
    (root / "model.py").write_text("ORIGINAL = True\n")
    workdir = tmp_path / "run"
    target_guard.record(workdir, root, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"])
    (root / "model.py").write_text("EDITED = True\n")
    (workdir / "emit.txt").write_text("Done.\n")
    with pytest.raises(ValueError, match="model.py"):
        agent_delivery.finish(workdir, "emit")


def test_a_target_that_cannot_be_read_is_a_warning_not_a_failed_step(target, tmp_path, monkeypatch):
    ctx = _prepare(tmp_path / "run")

    def unreadable(*args, **kwargs):
        raise ValueError("git cannot report the state of the target")

    monkeypatch.setattr(target_guard, "state", unreadable)
    assert "could not be compared" in (target_guard.warn_if_moved(ctx.workdir) or "")


def test_an_output_only_node_that_writes_only_its_output_is_delivered(target):
    workdir = target / "runs" / "one"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "emit", "Write emit.txt.", ["emit.txt"])
    (workdir / "emit.txt").write_text(render({"tool.py": "print('built')\n"}))
    assert agent_delivery.finish(workdir, "emit")["status"] == "delivered"


def test_a_node_that_is_not_output_only_is_not_held_to_its_outputs(target, tmp_path):
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit",))
    agent_delivery.begin(workdir, "research_director", "Research.", ["research-plan.json"])
    (target / "scratch.txt").write_text("a probe the research ran\n")
    (workdir / "research-plan.json").write_text("[]\n")
    assert agent_delivery.finish(workdir, "research_director")["status"] == "delivered"


def test_a_planning_step_registered_as_output_only_is_refused_when_it_changes_the_target(target, tmp_path):
    """A planning step as output-only: target file changes refused (#347)."""
    workdir = tmp_path / "run"
    target_guard.record(workdir, target, output_only=("emit", "design_doc"))
    agent_delivery.begin(workdir, "design_doc", "Write the design.", ["design.md"])
    (target / "app.py").write_text("EDITED\n")
    (workdir / "design.md").write_text("the design\n")
    with pytest.raises(ValueError, match=r"changed while agent 'design_doc' ran"):
        agent_delivery.finish(workdir, "design_doc")


class _EditingModel(FakeModel):
    """The #319 worker: edits the checkout, then answers that it is already done."""

    def __init__(self, answers: list[str], target: Path) -> None:
        super().__init__(answers)
        self.target = target

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
        if "Write the artifact `" in prompt:
            (self.target / "app.py").write_text("print('edited in place')\n")
        return answer


def test_a_design_run_whose_emit_edits_the_target_fails_before_materialise(target, tmp_path):
    model = _EditingModel([
        RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
        DESIGN_DOC_ANSWER,
        DESIGN_DOC_ANSWER,
        render({"tool.py": "print('built')\n"}), *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3],
    ], target)
    result = _without_worktree().run(
        tmp_path / "run", model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "a tool", "check": "python tool.py"},
    )
    assert result.exit_ok is False
    assert "changed while agent 'emit' ran" in (result.error or "")
    assert "app.py" in (result.error or "")
    assert not (target / repo_change.ARTIFACTS / NAME).exists(), "materialise never ran"


def test_a_refused_emit_is_redone_from_a_fresh_emit_txt(target, tmp_path):
    workdir = tmp_path / "run"
    refused = _without_worktree().run(
        workdir, verdicts={"gate_strategy": "proceed"},
        model=_EditingModel([RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
                             DESIGN_DOC_ANSWER,
                             DESIGN_DOC_ANSWER,
                             "All changes are already applied."], target),
        task={"name": NAME, "brief": "a tool", "check": "python tool.py"},
    )
    assert "changed while agent 'emit' ran" in (refused.error or "")
    _git(target, "checkout", "--", "app.py")

    redone = DesignMode.resume(workdir, model=StaleEmitModel([render({"tool.py": "print('redone')\n"}),
                                                              *REVIEW_ANSWERS[:3],
                                                              VERIFICATION_ANSWER,
                                                              REVIEW_ANSWERS[3], "VERDICT: PASS"]))
    assert redone.error is None
    # The redo's emission is derived from its workspace's diff: the file it wrote, whole.
    assert (workdir / "emit.txt").read_text() == "===FILE: tool.py===\nprint('redone')\n"
    assert (target / repo_change.ARTIFACTS / NAME / "tool.py").read_text() == "print('redone')\n"
