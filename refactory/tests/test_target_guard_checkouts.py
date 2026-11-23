"""Refactory's own checkout is guarded when it is not the run's target (dm-164).

An agent that runs a refactory tool with the contract's Python imports refactory from the
driver checkout, or from a pinned run's pin, and the tool writes there from any directory
(b32l, 2026-10-21). Each test pins one rule, on the walker (a scripted design run) and on
the spine's worker bridge (`worker_node.execute`, which every spine agent node runs). The
guard watches tracked files and HEAD, and every path under the checkout's package roots;
what a refusal does to the tracked files is `target_guard.CHECKOUT_POLICY`'s, and each
policy is tested here. A new path under a package root is always moved out.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from refactory_dsh import agent_delivery as ad
from refactory_dsh import target_guard, worktree
from refactory_dsh.model import DshModelClient
from _design_worktree_support import _answers, _design
from tests_support import FakeModel, apply_in_workspace, git, git_repo, render

#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")

@pytest.fixture(autouse=True)
def _restore_policy(monkeypatch):
    """The refusal tests below use the restore policy; the policy tests set their own."""
    monkeypatch.setattr(target_guard, "CHECKOUT_POLICY", "restore")


def test_the_default_policy_is_restore_if_clean():
    source = Path(target_guard.__file__).read_text()
    assert '\nCHECKOUT_POLICY = "restore-if-clean"\n' in source


#: A refactory tool that, like the generators, writes beside the package it is imported from.
GENERATOR = ("from pathlib import Path\n"
             "here = Path(__file__).resolve().parent\n"
             "(here / 'generated.json').write_text('{\"generated\": true}\\n')\n"
             "(here / 'mine.py').write_text('print(\"the agent rewrote this\")\\n')\n")

#: What the committed generated file holds before a generator rewrites it.
COMMITTED = "{}\n"


def _checkout(root: Path, *, dirty: bool = True) -> Path:
    """A refactory checkout: a git repository holding refactory/refactory_dsh with a generator.

    Its generated file is tracked, as refactory's generated files are; ``dirty`` leaves
    the person's own uncommitted edit to a tracked file and untracked notes in it.
    """
    git_repo(root)
    package = root / "refactory" / "refactory_dsh"
    package.mkdir(parents=True)
    (package / "gen.py").write_text(GENERATOR)
    (package / "mine.py").write_text("print('committed')\n")
    (package / "generated.json").write_text(COMMITTED)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "refactory")
    if dirty:
        # The person's own state there, which a restore must give back exactly.
        (package / "mine.py").write_text("print('the person, uncommitted')\n")
        (root / "notes.txt").write_text("the person's untracked notes\n")
    return root


def _generated(checkout: Path) -> str:
    return (checkout / "refactory" / "refactory_dsh" / "generated.json").read_text()


def _files(root: Path) -> dict[str, bytes]:
    found = {}
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name != ".git"]
        for name in files:
            path = Path(directory) / name
            found[str(path.relative_to(root))] = path.read_bytes()
    return found


def _contract(prompt: str) -> dict:
    block = prompt[prompt.index(ad.CONTRACT_START):prompt.index(ad.CONTRACT_END)]
    return json.loads(block[block.index("\n{") + 1:])


def _as_driver(monkeypatch, checkout: Path) -> None:
    """Make ``checkout`` the one this process, and so the contract's Python, imports refactory from."""
    monkeypatch.setattr(ad, "__file__", str(checkout / "refactory" / "refactory_dsh" / "agent_delivery.py"))


def _record(run: Path, *, checkout: Path, pin: Path | None = None) -> None:
    """What `refactory ceo` records at launch (`pin.record_driver`), and a pinned resume adds."""
    config = run / target_guard.RUN_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(config.read_text()) if config.exists() else {}
    saved["driver"] = {"checkout": str(checkout), "commit": git(checkout, "rev-parse", "HEAD"),
                       "dirty_diff": None, "fingerprints_at_launch": {}, "profile": "graph",
                       "pin_checkout": str(pin) if pin else None}
    config.write_text(json.dumps(saved))


class Emit(FakeModel):
    """A scripted run whose emit runs a refactory generator with the contract's Python."""

    def __init__(self, answers, *, generator=None, first=None):
        super().__init__(answers)
        self.generator, self.first, self.seen = generator, first, {}

    def complete(self, prompt, *, system="", workdir=None, **_workspace):
        if self.first is not None:
            self.first()
            self.first = None
        # A workspace node's cwd/env/confine (#516) go on to FakeModel, which makes the
        # scripted emission in the workspace.
        answer = super().complete(prompt, system=system, workdir=workdir, **_workspace)
        if ad.CONTRACT_START in prompt and _contract(prompt)["node"] == "emit" and "ran" not in self.seen:
            execution = _contract(prompt)
            self.seen.update(ran=True, prompt=prompt, execution=execution)
            generator = self.generator or Path(execution["refactory_source"]) / "gen.py"
            # From a scratch directory, as a builder's shell is: the tool writes where it lives.
            subprocess.run([sys.executable, str(generator)], cwd=workdir, check=True)
            self.seen["wrote"] = (Path(generator).parent / "generated.json").read_text() != COMMITTED
        return answer


# ── the walker: a scripted design run ─────────────────────────────────────────

def test_walker_emit_generator_writing_into_the_driver_is_refused_and_restored(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    _as_driver(monkeypatch, driver)
    repo = git_repo(tmp_path / "repo")
    before, target_before = _files(driver), _files(repo)
    model = Emit(_answers(render({"tool.py": "print('built')\n"})))

    result = _design(tmp_path / "run", model, target=str(repo))

    assert model.seen.get("wrote") is True, "the generator wrote into the driver"
    assert model.seen["execution"]["refactory_source"] == str(driver / "refactory" / "refactory_dsh")
    assert result.exit_ok is False
    assert f"refactory's own checkout {driver.resolve()} changed while agent 'emit' ran" in result.error
    assert "It is not this run's target" in result.error and str(repo.resolve()) in result.error
    assert _files(driver) == before, "the driver is back as it was, the person's edits included"
    assert _files(repo) == target_before
    receipt = ad.receipts_in(tmp_path / "run")["emit"]
    assert receipt["status"] == "failed"
    assert list(receipt["checkout_changes"]) == [str(driver.resolve())]
    kept = list((tmp_path / "run" / ad.ROOT / receipt["id"] / "checkouts").rglob("generated.json"))
    assert kept and kept[0].read_text() != COMMITTED, "the changed copy is kept, not lost"
    assert "The changed copies are in" in result.error


def test_walker_pinned_run_guards_its_pin(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    pin = _checkout(tmp_path / "pin")
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    before = _files(pin)
    # Pinned mid-run, as a resume after the driver moved records it (`pin.resolve_pin`).
    model = Emit(_answers(render({"tool.py": "print('built')\n"})),
                 generator=pin / "refactory" / "refactory_dsh" / "gen.py",
                 first=lambda: _record(run, checkout=driver, pin=pin))

    result = _design(run, model, target=str(repo))

    assert model.seen.get("wrote") is True
    assert result.exit_ok is False
    assert f"refactory's own checkout {pin.resolve()} changed while agent 'emit' ran" in result.error
    assert _files(pin) == before


def test_walker_run_whose_driver_is_its_target_is_unchanged(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    _as_driver(monkeypatch, driver)
    watched = []
    real = target_guard.watch_checkouts
    monkeypatch.setattr(target_guard, "watch_checkouts",
                        lambda workdir, paths, **kw: watched.append(list(paths)) or real(workdir, paths, **kw))
    before = _files(driver)
    model = Emit(_answers(render({"tool.py": "print('built')\n"})))

    result = _design(tmp_path / "run", model, target=str(driver))

    assert watched and all(paths == [] for paths in watched), watched
    # The target guard refuses and restores the write, as it did before dm-164; nothing else does.
    assert result.exit_ok is False
    assert "changed while agent 'emit' ran" in result.error
    assert "refactory's own checkout" not in result.error
    assert _files(driver) == before


def test_walker_legitimate_writes_pass_with_the_driver_watched(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    _as_driver(monkeypatch, driver)
    repo = git_repo(tmp_path / "repo")
    watched = []
    real = target_guard.watch_checkouts
    monkeypatch.setattr(target_guard, "watch_checkouts",
                        lambda workdir, paths, **kw: watched.append(list(paths)) or real(workdir, paths, **kw))

    class Scratch(FakeModel):
        def complete(self, prompt, *, system="", workdir=None, **_workspace):
            if workdir is not None:
                # An agent's own notes in its run directory are its to write.
                (Path(workdir) / "scratch-notes.txt").write_text(prompt[:40])
            answer = super().complete(prompt, system=system, workdir=workdir)
            # Design's emit works in a clone (#516): the scripted emission is made there.
            if _workspace.get("cwd") is not None:
                apply_in_workspace(answer, _workspace["cwd"])
            return answer

    # The run directory lies inside the driver checkout: it is still the run's own.
    result = _design(driver / "runs" / "one", Scratch(_answers(render({"tool.py": "print('built')\n"}))),
                     target=str(repo))

    assert result.exit_ok, result.error
    assert watched and all(paths == [driver.resolve()] for paths in watched), watched


# ── the spine: the worker bridge every spine agent node runs ──────────────────

def _spine_run(tmp_path: Path, rid: str) -> tuple[Path, Path, worktree.Worktree]:
    """A prepared worktree run, as design's prepare leaves it."""
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    found = worktree.create(repo, "HEAD", rid, run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))
    return repo, run, found


def _dispatch(run: Path, node: str, writes: list[str], monkeypatch, act) -> dict:
    """Run one agent node through `worker_node.execute`, the agent acting as ``act`` says."""
    from refactory_dsh.worker_node import execute

    def complete(self, prompt, *, system="", workdir=None, **_workspace):
        act(_contract(prompt), Path(workdir))
        return "Done."

    monkeypatch.setattr(DshModelClient, "complete", complete)
    client = DshModelClient(python_bin=Path(sys.executable).parent)
    return execute({"schema_version": 1, "node_id": node, "reads": [], "writes": writes, "produces": [],
                    "prompt": "Do the node's work.", "model_context": client.context()}, run)


def _generate_in(checkout: Path):
    def act(execution: dict, workdir: Path) -> None:
        (workdir / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
        subprocess.run([sys.executable, str(checkout / "refactory" / "refactory_dsh" / "gen.py")],
                       cwd=workdir, check=True)
    return act


def test_spine_emit_writing_into_the_recorded_driver_is_refused_and_restored(tmp_path, monkeypatch, capsys):
    driver = _checkout(tmp_path / "driver")
    repo, run, _ = _spine_run(tmp_path, "dm164spine01")
    _record(run, checkout=driver)
    before = _files(driver)

    with pytest.raises(ValueError, match=f"refactory's own checkout {driver.resolve()} changed while agent 'emit'") as refused:
        _dispatch(run, "emit", ["emit.txt"], monkeypatch, _generate_in(driver))

    assert _files(driver) == before
    receipt = ad.receipts_in(run)["emit"]
    assert receipt["status"] == "failed"
    assert not (run / "emit.txt").exists(), "the redo's worker must not find the refused answer"
    assert receipt["set_aside"] in str(refused.value)
    # The person is told, and the checkout keeps its own record of what was done to it.
    said = capsys.readouterr().err
    assert f"refactory: run warning: agent 'emit' changed refactory's own checkout {driver.resolve()}" in said
    logged = target_guard.restore_log(driver.resolve()).read_text()
    set_aside = str(run / ad.ROOT / receipt["id"] / "checkouts")
    assert "refactory/refactory_dsh/generated.json" in logged and set_aside in logged, logged


def test_spine_pinned_run_guards_its_pin(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    pin = _checkout(tmp_path / "pin")
    repo, run, _ = _spine_run(tmp_path, "dm164spine02")
    _record(run, checkout=driver, pin=pin)
    before = _files(pin)
    pythons = []

    def act(execution, workdir):
        pythons.append(execution["python"])
        _generate_in(pin)(execution, workdir)

    # Any agent node, not only emit: a reviewer that runs a tool is held to the same rule.
    with pytest.raises(ValueError, match=f"refactory's own checkout {pin.resolve()} changed while agent 'code_reviewer'"):
        _dispatch(run, "code_reviewer", ["emit.txt"], monkeypatch, act)

    assert pythons == [str((pin / ".venv" / "bin").resolve() / "python")], "the contract's Python is the pin's"
    assert _files(pin) == before


def test_spine_commit_into_the_driver_is_refused(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    repo, run, _ = _spine_run(tmp_path, "dm164spine05")
    _record(run, checkout=driver)

    def act(execution, workdir):
        (workdir / "reviews").mkdir(exist_ok=True)
        (workdir / "reviews" / "code.md").write_text("VERDICT: pass\n")
        (driver / "committed.py").write_text("print('committed by the agent')\n")
        git(driver, "add", "committed.py")
        git(driver, "commit", "-q", "-m", "agent commit")

    head = git(driver, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="committed files under it changed") as refused:
        _dispatch(run, "code_reviewer", ["reviews/code.md"], monkeypatch, act)

    assert f"refactory's own checkout {driver.resolve()}" in str(refused.value)
    # Not undone, and not claimed to be: the exact command that undoes it is named.
    assert f"`git -C {driver.resolve()} reset --keep {head}`" in str(refused.value)
    assert "put back" not in str(refused.value)
    assert (driver / "committed.py").exists()


def test_spine_run_whose_driver_is_its_target_is_unchanged(tmp_path, monkeypatch):
    repo, run, found = _spine_run(tmp_path, "dm164spine03")
    _record(run, checkout=repo)
    recorded = []
    real = target_guard.watch_checkouts
    monkeypatch.setattr(target_guard, "watch_checkouts",
                        lambda workdir, paths, **kw: recorded.append(list(paths)) or real(workdir, paths, **kw))

    def act(execution, workdir):
        (workdir / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
        (repo / "stray.py").write_text("print('into the target')\n")

    with pytest.raises(ValueError) as refused:
        _dispatch(run, "emit", ["emit.txt"], monkeypatch, act)

    assert "changed while agent 'emit' ran" in str(refused.value)
    assert "refactory's own checkout" not in str(refused.value)
    assert not (repo / "stray.py").exists()
    assert all(str(repo.resolve()) not in map(str, paths) for paths in recorded), recorded


def test_spine_legitimate_writes_to_the_run_and_its_worktree_pass(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    repo, run, found = _spine_run(tmp_path, "dm164spine04")
    _record(run, checkout=driver)

    def act(execution, workdir):
        (workdir / "reviews").mkdir(exist_ok=True)
        (workdir / "reviews" / "code.md").write_text("VERDICT: pass\n")
        (workdir / "scratch-notes.txt").write_text("notes\n")
        # A node that is not output-only may work in the run's worktree.
        (Path(found.path) / "probe.txt").write_text("probe\n")
        # The driver's .factory holds runs and their state, as a target's does.
        (driver / ".factory").mkdir(exist_ok=True)
        (driver / ".factory" / "state.json").write_text("{}\n")

    receipt = _dispatch(run, "code_reviewer", ["reviews/code.md"], monkeypatch, act)

    assert receipt["status"] == "delivered"
    assert "checkout_changes" not in receipt


# ── the bracket's edges, on the route every runtime shares ────────────────────

def _bracket_run(tmp_path: Path, **record) -> tuple[Path, Path, Path]:
    driver = _checkout(tmp_path / "driver")
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    target_guard.record(run, repo, output_only=("emit",), **record)
    _record(run, checkout=driver)
    return driver, repo, run


def test_a_parallel_node_that_saw_the_stray_is_not_refused_for_its_undoing(tmp_path):
    driver, _, run = _bracket_run(tmp_path)
    (run / "reviews").mkdir(parents=True)
    ad.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (driver / "app.py").write_text("print('stray')\n")
    # qa_director starts while the stray edit is in the driver.
    ad.begin(run, "qa_director", "QA.", ["reviews/qa.md"])
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    with pytest.raises(ValueError, match="refactory's own checkout .* changed while agent 'code_reviewer'"):
        ad.finish(run, "code_reviewer")
    assert (driver / "app.py").read_text() == "print('app')\n"
    (run / "reviews" / "qa.md").write_text("VERDICT: pass\n")
    assert ad.finish(run, "qa_director")["status"] == "delivered"


def test_a_retried_attempt_is_judged_from_a_fresh_snapshot(tmp_path):
    """A refused attempt whose record is still pending (a bridge that died before settling it)."""
    driver, _, run = _bracket_run(tmp_path)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    pending = ad._pending(run, "emit")
    kept = pending.read_text()
    (driver / "app.py").write_text("print('stray')\n")
    (run / "emit.txt").write_text("Done.\n")
    with pytest.raises(ValueError, match="refactory's own checkout"):
        ad.finish(run, "emit")
    pending.write_text(kept)
    # The person changes their own checkout between the attempts, and keeps it.
    mine = driver / "refactory" / "refactory_dsh" / "mine.py"
    mine.write_text("print('the person, later')\n")
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    assert ad.finish(run, "emit")["status"] == "delivered"
    assert mine.read_text() == "print('the person, later')\n"


def test_a_retried_node_with_only_a_checkout_snapshot_is_judged_afresh(tmp_path):
    """A node the target guard does not watch (not output-only, no worktree) still retries afresh."""
    driver, _, run = _bracket_run(tmp_path)
    (run / "reviews").mkdir(parents=True)
    record = ad.begin(run, "code_reviewer", "Review.", ["reviews/code.md"], reuse=True)
    assert record["target"] is None and record["checkouts"], record
    pending = ad._pending(run, "code_reviewer")
    kept = pending.read_text()
    (driver / "app.py").write_text("print('stray')\n")
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    with pytest.raises(ValueError, match="refactory's own checkout"):
        ad.finish(run, "code_reviewer")
    pending.write_text(kept)
    mine = driver / "refactory" / "refactory_dsh" / "mine.py"
    mine.write_text("print('the person, later')\n")
    ad.begin(run, "code_reviewer", "Review.", ["reviews/code.md"], reuse=True)
    (run / "reviews" / "code.md").write_text("VERDICT: pass, again\n")
    assert ad.finish(run, "code_reviewer")["status"] == "delivered"


def test_a_launcher_that_is_the_driver_is_refused_not_only_warned_about(tmp_path, capsys):
    driver = tmp_path / "driver"
    driver, _, run = _bracket_run(tmp_path, warn_also=(driver,))
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (driver / "app.py").write_text("print('stray')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="refactory's own checkout"):
        ad.finish(run, "emit")
    assert "files outside its worktree changed" not in capsys.readouterr().err


def test_untracked_files_in_the_checkout_are_not_watched(tmp_path):
    """An editor's swap file, a log appended through an open handle, a new notes file."""
    driver, _, run = _bracket_run(tmp_path)
    log = (driver / "run.log").open("a")
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (driver / ".mine.py.swp").write_bytes(b"swap")
    log.write("a line\n")
    log.flush()
    (driver / "notes.txt").write_text("the person's notes, later\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    assert ad.finish(run, "emit")["status"] == "delivered"
    log.close()
    assert (driver / ".mine.py.swp").exists() and (driver / "notes.txt").read_text().endswith("later\n")


def test_a_file_whose_bytes_were_not_kept_is_left_where_it_is(tmp_path, monkeypatch, capsys):
    driver, _, run = _bracket_run(tmp_path)
    # Its uncommitted bytes are larger than a baseline keeps.
    monkeypatch.setattr(target_guard, "MAX_BASELINE_SIZE", 8)
    mine = driver / "refactory" / "refactory_dsh" / "mine.py"
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    assert "cannot be put back if this node changes them: refactory/refactory_dsh/mine.py" in capsys.readouterr().err
    mine.write_text("print('changed during the node')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="refactory/refactory_dsh/mine.py \\(its content before the node was not kept; left as it is\\)"):
        ad.finish(run, "emit")
    assert mine.read_text() == "print('changed during the node')\n", "never moved away with nothing to put back"


def test_a_checkout_git_cannot_read_afterwards_is_warned_about_not_fatal(tmp_path, capsys):
    driver, _, run = _bracket_run(tmp_path)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (driver / ".git").rename(driver / ".git-away")
    try:
        (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
        assert ad.finish(run, "emit")["status"] == "delivered"
    finally:
        (driver / ".git-away").rename(driver / ".git")
    assert "whether the node changed it is unknown" in capsys.readouterr().err


# ── the restore policy: one setting, each choice tested ───────────────────────

def _stray_edit(tmp_path: Path, monkeypatch, policy: str, *, dirty: bool) -> tuple[Path, str]:
    """An emit that edits a tracked file in the driver, under ``policy``; the driver and the refusal."""
    monkeypatch.setattr(target_guard, "CHECKOUT_POLICY", policy)
    driver = _checkout(tmp_path / "driver", dirty=dirty)
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    target_guard.record(run, repo, output_only=("emit",))
    _record(run, checkout=driver)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (driver / "app.py").write_text("print('the agent edited this')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="refactory's own checkout") as refused:
        ad.finish(run, "emit")
    return driver, str(refused.value)


def test_policy_restore_puts_the_files_back(tmp_path, monkeypatch):
    driver, refusal = _stray_edit(tmp_path, monkeypatch, "restore", dirty=True)
    assert (driver / "app.py").read_text() == "print('app')\n"
    assert "put back as they were before the node: app.py" in refusal


def test_policy_report_leaves_the_files_and_names_them_with_their_diff(tmp_path, monkeypatch, capsys):
    driver, refusal = _stray_edit(tmp_path, monkeypatch, "report", dirty=False)
    assert (driver / "app.py").read_text() == "print('the agent edited this')\n"
    assert "They were left as they are: app.py." in refusal
    said = capsys.readouterr().err
    assert "left as they are: app.py" in said and "+print('the agent edited this')" in said
    assert "left as they are: app.py" in target_guard.restore_log(driver.resolve()).read_text()


def test_policy_report_shows_only_what_the_node_changed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(target_guard, "CHECKOUT_POLICY", "report")
    driver, _, run = _bracket_run(tmp_path)
    mine = driver / "refactory" / "refactory_dsh" / "mine.py"
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    mine.write_text("print('the agent, during the node')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="left as they are: refactory/refactory_dsh/mine.py"):
        ad.finish(run, "emit")
    said = capsys.readouterr().err
    # From the person's uncommitted bytes at the node's start, not from HEAD.
    assert "-print('the person, uncommitted')" in said and "+print('the agent, during the node')" in said
    assert "-print('committed')" not in said


def test_policy_restore_if_clean_restores_a_clean_checkout(tmp_path, monkeypatch):
    driver, refusal = _stray_edit(tmp_path, monkeypatch, "restore-if-clean", dirty=False)
    assert (driver / "app.py").read_text() == "print('app')\n"


def test_policy_restore_if_clean_reports_on_a_checkout_the_person_was_editing(tmp_path, monkeypatch):
    driver, refusal = _stray_edit(tmp_path, monkeypatch, "restore-if-clean", dirty=True)
    assert (driver / "app.py").read_text() == "print('the agent edited this')\n"
    assert "They were left as they are: app.py." in refusal


def test_an_unknown_policy_is_refused(monkeypatch):
    monkeypatch.setattr(target_guard, "CHECKOUT_POLICY", "rewind")
    with pytest.raises(ValueError, match="must be one of restore, report, restore-if-clean"):
        target_guard.restores({"changes": {}})


# ── which checkouts are watched ───────────────────────────────────────────────

def test_checkouts_skip_the_target_its_worktree_and_anything_not_a_checkout(tmp_path):
    driver = _checkout(tmp_path / "driver")
    repo = git_repo(tmp_path / "repo")
    run = tmp_path / "run"
    source = driver / "refactory" / "refactory_dsh"
    assert target_guard.checkouts(run, source) == [], "a run that recorded no target is not watched"
    target_guard.record(run, repo)
    assert target_guard.checkouts(run, source) == [driver.resolve()]
    target_guard.record(run, repo, also=(driver,))
    assert target_guard.checkouts(run, source) == [], "the run's own worktree is never refactory's"
    target_guard.record(run, driver / "refactory")
    assert target_guard.checkouts(run, source) == [], "a target inside the driver is the target guard's"
    target_guard.record(run, repo)
    plain = tmp_path / "installed" / "refactory_dsh"
    plain.mkdir(parents=True)
    assert target_guard.checkouts(run, plain) == [], "an installed package is no checkout"
    (driver / ".gitignore").write_text(".venv/\n")
    installed = driver / ".venv" / "lib" / "refactory_dsh"
    installed.mkdir(parents=True)
    assert target_guard.checkouts(run, installed) == [], "nor is one installed where a checkout ignores it"
    (run / target_guard.RUN_CONFIG).write_text(json.dumps({"driver": {"checkout": str(plain.parent)}}))
    assert target_guard.checkouts(run, plain) == [], "nor is a recorded driver that is not a checkout"


def test_the_suite_never_watches_its_own_checkout(tmp_path, monkeypatch):
    from refactory_dsh.paths import REPO_ROOT

    run = tmp_path / "run"
    target_guard.record(run, git_repo(tmp_path / "repo"))
    source = Path(ad.__file__).resolve().parent
    assert target_guard.checkouts(run, source) == [], "conftest names this checkout as unwatched"
    monkeypatch.delenv(target_guard.UNWATCHED_ENV)
    assert target_guard.checkouts(run, source) == [REPO_ROOT.resolve()]


def test_the_unwatched_variable_is_ignored_outside_a_test(tmp_path, monkeypatch, capsys):
    """A person's shell never has PYTEST_CURRENT_TEST: there the variable must not silence the guard."""
    driver, _, run = _bracket_run(tmp_path)
    monkeypatch.setenv(target_guard.UNWATCHED_ENV, str(driver))
    monkeypatch.setattr(target_guard, "_IGNORED_UNWATCHED", [])
    source = driver / "refactory" / "refactory_dsh"
    assert target_guard.checkouts(run, source) == [], "inside a test it is honoured"
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert target_guard.checkouts(run, source) == [driver.resolve()]
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (driver / "app.py").write_text("print('stray')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="refactory's own checkout"):
        ad.finish(run, "emit")
    said = capsys.readouterr().err
    warned = [line for line in said.splitlines()
              if line.startswith(f"refactory: warning: {target_guard.UNWATCHED_ENV} is set")]
    assert len(warned) == 1, said


def test_a_checkout_is_read_from_one_status_call(tmp_path):
    driver = _checkout(tmp_path / "driver")
    # An old name that reads like a status entry of its own.
    odd = "1 2 3 4 5 6 7 8 odd.py"
    (driver / odd).write_text("print('odd')\n")
    git(driver, "add", odd)
    git(driver, "commit", "-q", "-m", "odd")
    git(driver, "mv", odd, "renamed odd.py")
    (driver / "new dir").mkdir()
    (driver / "new dir" / "u.txt").write_text("untracked\n")
    found = target_guard._checkout_state(driver)
    assert found["head"] == found["tree"] == git(driver, "rev-parse", "HEAD")
    # Tracked files only, and a rename names its new path only, however its old one reads.
    assert sorted(found["changes"]) == ["refactory/refactory_dsh/mine.py", "renamed odd.py"]
    assert target_guard._porcelain_v2("# branch.oid (initial)\0")[0] is None


# ── new paths under the checkout's package roots (create benchmark H1/H2, C1) ─────────
#
# A create run's emit agent wrote a test copy of its package into the driver's
# `.factory/packages/`, and an overwatch agent linked one into `refactory/refactory_packages/`
# (2026-11-20). Neither path is tracked, so only the package-root watch sees them.

def _create_run(tmp_path: Path, *, project: Path | None = None) -> tuple[Path, Path, Path]:
    """A create run as `prepare_create` leaves it: the project recorded, the driver named."""
    driver = _checkout(tmp_path / "driver")
    project = project or (tmp_path / "project")
    project.mkdir(parents=True, exist_ok=True)
    run = tmp_path / "run"
    target_guard.record(run, project)
    _record(run, checkout=driver)
    return driver, project, run


def _package(root: Path, name: str) -> Path:
    package = root / ".factory" / "packages" / name
    package.mkdir(parents=True)
    for file in ("__init__.py", "mode.py", "ops.py"):
        (package / file).write_text(f"# {file}\n")
    return package


def test_a_package_copy_written_into_the_driver_is_refused_and_left_in_place(tmp_path, capsys):
    driver, _project, run = _create_run(tmp_path)
    (driver / ".factory" / "packages").mkdir(parents=True)
    lock = driver / ".factory" / "packages" / ".created_pkg.lock"
    lock.write_text("")
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    _package(driver, "h1_pkg")  # what H1's emit agent did to import its package
    (run / "emit.txt").write_text("===FILE: __init__.py===\n")
    with pytest.raises(ValueError, match="new paths under its package roots: .factory/packages/h1_pkg") as refused:
        ad.finish(run, "emit")
    assert (driver / ".factory" / "packages" / "h1_pkg" / "ops.py").exists(), "refactory moves nothing"
    assert lock.exists()
    text, said = str(refused.value), capsys.readouterr().err
    assert "A node must not write under refactory's package roots" in text
    assert "if a person or another tool made them, nothing needs doing" in text
    assert text.count(".factory/packages/h1_pkg") == 1, "named once"
    for reader in (text, said):  # a CEO agent reads both: nothing tells it to delete
        assert "rm " not in reader and "remove them" not in reader.lower()


def test_a_package_symlink_into_the_driver_is_refused_on_the_spine(tmp_path, monkeypatch):
    driver, project, run = _create_run(tmp_path)
    mine = _package(project, "h2_pkg")
    installed = driver / "refactory" / "refactory_packages"
    installed.mkdir(parents=True)

    def act(execution, workdir):  # what H2's overwatch agent did, through the spine's worker bridge
        (workdir / "overwatch.md").write_text("ok\n")
        os.symlink(mine, installed / "h2_pkg")

    with pytest.raises(ValueError, match="new paths under its package roots: refactory/refactory_packages/h2_pkg"):
        _dispatch(run, "overwatch", ["overwatch.md"], monkeypatch, act)
    assert os.path.islink(installed / "h2_pkg"), "reported, never moved"
    assert sorted(p.name for p in mine.iterdir()) == ["__init__.py", "mode.py", "ops.py"]


def test_a_path_already_under_the_package_roots_is_reported_not_reverted(tmp_path):
    driver, _project, run = _create_run(tmp_path)
    old = _package(driver, "old_pkg")
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (old / "ops.py").write_text("# rewritten by the node, longer\n")
    (old / "new.py").write_text("# new\n")
    (run / "emit.txt").write_text("===FILE: __init__.py===\n")
    with pytest.raises(ValueError) as refused:
        ad.finish(run, "emit")
    text = str(refused.value)
    assert "paths that were already under its package roots changed: .factory/packages/old_pkg/ops.py" in text
    assert "left as they are: .factory/packages/old_pkg/ops.py" in text
    assert (old / "ops.py").read_text() == "# rewritten by the node, longer\n", "never reverted"
    assert "new paths under its package roots: .factory/packages/old_pkg/new.py" in text
    assert (old / "new.py").exists(), "a new path inside it is reported, not removed"


def test_importing_a_package_in_the_driver_is_not_a_write(tmp_path):
    driver, _project, run = _create_run(tmp_path)
    installed = driver / "refactory" / "refactory_packages" / "fix_pack"
    installed.mkdir(parents=True)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (installed / "__pycache__").mkdir()
    (installed / "__pycache__" / "ops.cpython-312.pyc").write_bytes(b"pyc")
    (installed / ".DS_Store").write_bytes(b"finder")
    (run / "emit.txt").write_text("===FILE: __init__.py===\n")
    assert ad.finish(run, "emit")["status"] == "delivered"


def test_a_project_that_is_a_refactory_clone_may_write_its_own_packages(tmp_path):
    project = _checkout(tmp_path / "clone")
    driver, project, run = _create_run(tmp_path, project=project)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    _package(project, "its_own")
    (project / "refactory" / "refactory_packages").mkdir(parents=True)
    os.symlink(project / ".factory" / "packages" / "its_own",
               project / "refactory" / "refactory_packages" / "its_own")
    (run / "emit.txt").write_text("===FILE: __init__.py===\n")
    assert ad.finish(run, "emit")["status"] == "delivered"
    assert (project / ".factory" / "packages" / "its_own" / "ops.py").exists()
    assert target_guard.checkouts(run, tmp_path / "elsewhere") == [driver.resolve()], \
        "the project is never watched as refactory's own, the driver is"


def test_a_run_whose_project_is_the_driver_does_not_watch_it(tmp_path):
    driver = _checkout(tmp_path / "driver")
    run = tmp_path / "run"
    target_guard.record(run, driver)
    _record(run, checkout=driver)
    assert target_guard.checkouts(run, driver / "refactory" / "refactory_dsh") == []


def test_package_entries_are_paths_under_the_package_roots_only(tmp_path):
    driver = _checkout(tmp_path / "driver")
    _package(driver, "p")
    (driver / "notes.txt").write_text("elsewhere\n")
    found = target_guard._checkout_state(driver)["packages"]
    assert sorted(found) == [".factory/packages/p", ".factory/packages/p/__init__.py",
                             ".factory/packages/p/mode.py", ".factory/packages/p/ops.py"]
    assert target_guard.package_differences(found, dict(found)) == ([], [])
    added, _ = target_guard.package_differences({".factory/packages/p": "directory"}, found)
    assert added == [".factory/packages/p/__init__.py", ".factory/packages/p/mode.py",
                     ".factory/packages/p/ops.py"], "a new file in an old directory is new"
    added, _ = target_guard.package_differences({}, found)
    assert added == [".factory/packages/p"], "only the top of a new tree"


# ── the reviewer's cases (PR 722): what is not the node's is never charged to it ──────

def test_a_pulled_package_that_git_tracks_is_not_the_nodes(tmp_path):
    """The person pulls a commit adding a package while a create node runs."""
    driver, _project, run = _create_run(tmp_path)
    git(driver, "stash", "-u")
    git(driver, "checkout", "-q", "-b", "feature")
    pulled = driver / "refactory" / "refactory_packages" / "newpkg"
    pulled.mkdir(parents=True)
    (pulled / "__init__.py").write_text("x = 1\n")
    git(driver, "add", ".")
    git(driver, "commit", "-q", "-m", "add pkg")
    git(driver, "checkout", "-q", "-")
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    git(driver, "merge", "-q", "--ff-only", "feature")
    (pulled / "junk.py").write_text("# the node's\n")
    (run / "emit.txt").write_text("ok\n")
    with pytest.raises(ValueError) as refused:  # its committed files moved: the tracked check's
        ad.finish(run, "emit")
    text = str(refused.value)
    assert "new paths under its package roots: refactory/refactory_packages/newpkg/junk.py" in text
    assert "refactory_packages/newpkg," not in text and "newpkg/__init__.py" not in text
    assert (pulled / "__init__.py").exists() and (pulled / "junk.py").exists()
    assert git(driver, "status", "--short", "--untracked-files=no") == ""


def test_a_package_another_watcher_could_have_written_is_a_warning_naming_both(tmp_path, capsys):
    """Two nodes watch the driver at once; a package appears: neither is refused for it."""
    driver, _project, run = _create_run(tmp_path)
    (run / "reviews").mkdir(parents=True)
    ad.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    ad.begin(run, "qa_director", "QA.", ["reviews/qa.md"])
    _package(driver, "whose_pkg")
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    (run / "reviews" / "qa.md").write_text("VERDICT: pass\n")
    assert ad.finish(run, "code_reviewer")["status"] == "delivered"
    assert ad.finish(run, "qa_director")["status"] == "delivered"
    said = capsys.readouterr().err
    assert "changed while more than one node watched it" in said
    assert f"qa_director in {run}" in said and f"code_reviewer in {run}" in said
    assert (driver / ".factory" / "packages" / "whose_pkg").exists()


def test_a_watch_that_ended_before_the_node_began_does_not_shield_it(tmp_path):
    driver, _project, run = _create_run(tmp_path)
    (run / "reviews").mkdir(parents=True)
    ad.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    assert ad.finish(run, "code_reviewer")["status"] == "delivered"
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    _package(driver, "h1_pkg")
    (run / "emit.txt").write_text("ok\n")
    with pytest.raises(ValueError, match="new paths under its package roots"):
        ad.finish(run, "emit")


def test_a_package_written_into_the_driver_by_someone_else_is_never_removed(tmp_path):
    """`refactory adopt` in the driver, or another run whose project is the driver."""
    driver, _project, run = _create_run(tmp_path)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    _package(driver, "persons_pkg")
    (run / "emit.txt").write_text("ok\n")
    with pytest.raises(ValueError):
        ad.finish(run, "emit")
    assert (driver / ".factory" / "packages" / "persons_pkg" / "ops.py").exists()


def test_a_differently_cased_driver_is_refused_as_a_project(tmp_path):
    from refactory_dsh import paths

    driver = _checkout(tmp_path / "driver")
    variant = Path(str(driver).replace("/driver", "/DRIVER"))
    if not variant.exists():
        pytest.skip("case-sensitive filesystem")
    run = tmp_path / "run"
    _record(run, checkout=driver)
    with pytest.raises(paths.CheckoutProjectError):
        paths.project_for_writes(variant, workdir=run)
    recorded = target_guard.record(run, variant)
    assert Path(recorded["root"]).samefile(driver)
    assert target_guard.checkouts(run, tmp_path / "elsewhere") == []


def test_too_many_package_paths_to_watch_is_said(tmp_path, monkeypatch, capsys):
    driver, _project, run = _create_run(tmp_path)
    _package(driver, "big")
    monkeypatch.setattr(target_guard, "LIMIT", 2)
    ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    assert "paths under its package roots, so a package an agent writes there is not watched" in \
        capsys.readouterr().err


def test_a_redo_closes_its_own_earlier_watch(tmp_path):
    """An attempt whose bridge died never finished its watch; its redo is not shielded by it."""
    driver, _project, run = _create_run(tmp_path)
    watched = [driver.resolve()]
    target_guard.watch_checkouts(run, watched, node="emit")
    again = target_guard.watch_checkouts(run, watched, node="emit")
    _package(driver, "h1_pkg")
    changed = target_guard.strayed_checkouts(run, again)
    assert "new paths under its package roots: .factory/packages/h1_pkg" in changed[str(driver.resolve())]



# ── watches belong to a process (PR 722, round 2) ─────────────────────────────────────

def test_a_killed_watchers_watch_stops_counting_as_an_overlap(tmp_path):
    """A run killed mid-node never finishes its watch; a later lone writer is still charged."""
    import signal

    driver, project, run = _create_run(tmp_path)
    script = ("import os, signal, sys\n"
              "from pathlib import Path\n"
              "from refactory_dsh import agent_delivery as ad\n"
              "ad.begin(Path(sys.argv[1]), 'crashed', 'Write c.txt.', ['c.txt'])\n"
              "os.kill(os.getpid(), signal.SIGKILL)\n")
    killed = subprocess.run([sys.executable, "-c", script, str(run)], capture_output=True, text=True,
                            timeout=120, check=False)
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    later = tmp_path / "run2"
    target_guard.record(later, project)
    _record(later, checkout=driver)
    ad.begin(later, "emit", "Write emit.txt.", ["emit.txt"])
    _package(driver, "stray_by_emit")
    (later / "emit.txt").write_text("ok\n")
    with pytest.raises(ValueError, match="new paths under its package roots: .factory/packages/stray_by_emit"):
        ad.finish(later, "emit")


def test_a_watch_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    driver = _checkout(tmp_path / "driver")
    run = tmp_path / "run"
    first = target_guard._open_watch(driver, run, "emit", "2026-11-22T00:00:00+00:00")
    def fail(*_args):
        raise OSError("disk full")
    monkeypatch.setattr(target_guard.os, "replace", fail)
    with pytest.raises(OSError):
        target_guard._close_watch(driver, first, "2026-11-22T00:01:00+00:00")
    assert target_guard._read_watch(driver, first)["finished"] is None, "the old record is whole"


def test_a_reused_dispatch_record_is_owned_by_the_process_that_carries_it_on(tmp_path):
    driver = _checkout(tmp_path / "driver")
    run = tmp_path / "run"
    watched = target_guard.watch_checkouts(run, [driver.resolve()], node="emit")
    identifier = watched[str(driver.resolve())]["watch_id"]
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                          text=True, check=True)
    dead = {**target_guard._read_watch(driver.resolve(), identifier), "pid": int(gone.stdout), "pid_start": "then"}
    target_guard._write_watch(driver.resolve(), dead)
    assert target_guard._gone(dead) is True, "setup: its first process is gone"
    target_guard.claim_watches(watched)
    assert target_guard._gone(target_guard._read_watch(driver.resolve(), identifier)) is False


def test_an_open_watch_nothing_can_check_lapses_after_the_backstop(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    driver = _checkout(tmp_path / "driver")
    run = tmp_path / "run"
    opened = "2026-11-22T00:00:00+00:00"
    # Read on a frozen clock a minute after both opened, whatever today's date: the live
    # one is far inside its backstop, the unchecked one past a WATCH_OPEN_S of 0. (Read on
    # the real clock, this fixed date lapsed both from 2026-11-23.)
    monkeypatch.setattr(target_guard, "_now", lambda: datetime.fromisoformat(opened) + timedelta(minutes=1))
    mine = target_guard._open_watch(driver, run, "here", opened)
    elsewhere = target_guard._open_watch(driver, run, "there", opened)
    target_guard._write_watch(driver, {**target_guard._read_watch(driver, elsewhere), "host": "another-host"})
    monkeypatch.setattr(target_guard, "WATCH_OPEN_S", 0)
    left = {watch["id"] for watch in target_guard._read_watches(driver)}
    assert left == {mine}, "a live process's watch stays open; one nobody can check lapses"


def test_a_dispatch_reused_after_its_bridge_died_keeps_its_watch_open(tmp_path):
    """The bridge that began the node is killed; the redo reuses the pending record."""
    import signal

    driver, _project, run = _create_run(tmp_path)
    script = ("import os, signal, sys\n"
              "from pathlib import Path\n"
              "from refactory_dsh import agent_delivery as ad\n"
              "ad.begin(Path(sys.argv[1]), 'emit', 'Write emit.txt.', ['emit.txt'], reuse=True)\n"
              "os.kill(os.getpid(), signal.SIGKILL)\n")
    killed = subprocess.run([sys.executable, "-c", script, str(run)], capture_output=True, text=True,
                            timeout=120, check=False)
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    record = ad.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    watch = target_guard._read_watch(driver.resolve(), record["checkouts"][str(driver.resolve())]["watch_id"])
    assert watch["pid"] == os.getpid() and target_guard._gone(watch) is False


# ── the interactive spine's bridge (PR 722, round 3) ──────────────────────────────────
#
# The spine runs `python -m refactory_dsh.agent_delivery` for `begin` and again for
# `finish`; the node runs on in the spine between them. Its watch belongs to the process
# the spine names (`REFACTORY_WATCH_OWNER_PID`), never to the short-lived command.

def _bridge(run: Path, request: dict, *, owner: int | None) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key != target_guard.OWNER_ENV}
    if owner is not None:
        env[target_guard.OWNER_ENV] = str(owner)
    return subprocess.run([sys.executable, "-m", "refactory_dsh.agent_delivery"], cwd=str(run), env=env,
                          input=json.dumps(request), capture_output=True, text=True, timeout=120, check=False)


def test_a_bridged_writer_overlapping_an_in_process_node_is_warned_not_refused(tmp_path, capsys):
    driver, _project, run = _create_run(tmp_path)
    begun = _bridge(run, {"action": "begin", "node": "writer", "prompt": "Write writer.txt.",
                          "writes": ["writer.txt"]}, owner=os.getpid())  # this test stands in for the spine
    assert begun.returncode == 0, begun.stderr
    ad.begin(run, "innocent", "Write innocent.txt.", ["innocent.txt"])
    _package(driver, "by_writer")  # the bridged node writes it, while the in-process node runs
    (run / "innocent.txt").write_text("ok\n")
    assert ad.finish(run, "innocent")["status"] == "delivered", "the in-process node is not blamed"
    said = capsys.readouterr().err
    assert "not charged to any one node" in said and f"writer in {run}" in said
    (run / "writer.txt").write_text("ok\n")
    finished = _bridge(run, {"action": "finish", "node": "writer"}, owner=os.getpid())
    assert finished.returncode == 0, finished.stderr
    assert json.loads(finished.stdout)["status"] == "delivered"
    assert "not charged to any one node" in finished.stderr


def test_the_bridge_records_the_owner_the_spine_names_never_itself(tmp_path):
    driver, _project, run = _create_run(tmp_path)
    # "named" twice: the spine reports a due node again, and the reused record is claimed.
    for node, owner in (("named", os.getpid()), ("named", os.getpid()), ("unnamed", None)):
        begun = _bridge(run, {"action": "begin", "node": node, "prompt": "Write it.", "writes": [f"{node}.txt"]},
                        owner=owner)
        assert begun.returncode == 0, begun.stderr
    watches = {watch["node"]: watch for watch in target_guard._read_watches(driver.resolve())}
    assert watches["named"]["pid"] == os.getpid() and target_guard._gone(watches["named"]) is False
    assert watches["unnamed"]["pid"] is None, "an unnamed owner is unknown, not the exited command"
    assert target_guard._gone(watches["unnamed"]) is None


def test_the_spine_names_itself_as_the_owner_of_bridged_watches():
    from refactory_dsh.paths import REPO_ROOT

    spine = (REPO_ROOT / "packages" / "workflow" / "workflow-spine" / "src" / "index.ts").read_text()
    bridge = spine[spine.index("function runtimeBridge("):]
    bridge = bridge[:bridge.index("\n  }\n")]
    assert f"{target_guard.OWNER_ENV}: String(process.pid)" in bridge


# ── a node that ends without a finish ends its watch (PR 722, round 4) ─────────────────

def _lone_writer(tmp_path: Path, driver: Path, project: Path, name: str):
    """Another run's node that writes a package into the driver on its own."""
    later = tmp_path / f"run-{name}"
    target_guard.record(later, project)
    _record(later, checkout=driver)
    ad.begin(later, "emit", "Write emit.txt.", ["emit.txt"])
    _package(driver, name)
    (later / "emit.txt").write_text("ok\n")
    return later


def test_an_invalidated_bridged_node_no_longer_shields_a_lone_writer(tmp_path):
    driver, project, run = _create_run(tmp_path)
    begun = _bridge(run, {"action": "begin", "node": "abandoned", "prompt": "Write it.",
                          "writes": ["abandoned.txt"]}, owner=os.getpid())  # the spine lives on
    assert begun.returncode == 0, begun.stderr
    skipped = _bridge(run, {"action": "invalidate", "nodes": ["abandoned"], "reason": "skip abandoned"},
                      owner=os.getpid())
    assert skipped.returncode == 0, skipped.stderr
    later = _lone_writer(tmp_path, driver, project, "by_lone_writer")
    with pytest.raises(ValueError, match="new paths under its package roots: .factory/packages/by_lone_writer"):
        ad.finish(later, "emit")


@pytest.mark.parametrize("status, refused", [("cancelled", True), ("failed", True), ("halted", True),
                                             ("ready", False)])
def test_a_stopped_runs_open_watch_no_longer_shields_a_lone_writer(tmp_path, monkeypatch, status, refused):
    driver, project, run = _create_run(tmp_path)
    assert _bridge(run, {"action": "begin", "node": "stopped", "prompt": "Write it.",
                         "writes": ["stopped.txt"]}, owner=os.getpid()).returncode == 0
    (run / ".refactory" / "spine-status.json").write_text(json.dumps({"status": status}))
    monkeypatch.setattr(target_guard, "WATCH_SETTLING_S", 0)
    later = _lone_writer(tmp_path, driver, project, "by_lone_writer")
    if refused:
        with pytest.raises(ValueError, match="new paths under its package roots"):
            ad.finish(later, "emit")
    else:
        assert ad.finish(later, "emit")["status"] == "delivered", "a running run's node still overlaps"


def test_a_watch_whose_delivery_record_is_gone_is_over(tmp_path, monkeypatch):
    driver, project, run = _create_run(tmp_path)
    ad.begin(run, "dropped", "Write it.", ["dropped.txt"])
    ad._pending(run, "dropped").unlink()  # its delivery ended by a path that closed nothing
    monkeypatch.setattr(target_guard, "WATCH_SETTLING_S", 0)
    later = _lone_writer(tmp_path, driver, project, "by_lone_writer")
    with pytest.raises(ValueError, match="new paths under its package roots"):
        ad.finish(later, "emit")


def test_a_watch_older_than_any_node_may_run_is_over_with_its_owner_alive(tmp_path):
    from datetime import datetime, timedelta, timezone

    driver, project, run = _create_run(tmp_path)
    record = ad.begin(run, "stuck", "Write it.", ["stuck.txt"])
    identifier = record["checkouts"][str(driver.resolve())]["watch_id"]
    long_ago = datetime.now(timezone.utc) - timedelta(seconds=target_guard._max_open_s() + 60)
    target_guard._write_watch(driver.resolve(), {**target_guard._read_watch(driver.resolve(), identifier),
                                                 "started": long_ago.isoformat()})
    assert target_guard._gone(target_guard._read_watch(driver.resolve(), identifier)) is False
    later = _lone_writer(tmp_path, driver, project, "by_lone_writer")
    with pytest.raises(ValueError, match="new paths under its package roots"):
        ad.finish(later, "emit")


def test_a_watch_just_opened_counts_before_its_delivery_record_exists(tmp_path, capsys):
    driver, project, run = _create_run(tmp_path)
    target_guard.watch_checkouts(run, [driver.resolve()], node="starting")  # its record comes next
    later = _lone_writer(tmp_path, driver, project, "by_lone_writer")
    assert ad.finish(later, "emit")["status"] == "delivered"
    assert "not charged to any one node" in capsys.readouterr().err
