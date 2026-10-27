"""Refactory's own checkout is guarded when it is not the run's target (dm-164).

An agent that runs a refactory tool with the contract's Python imports refactory from the
driver checkout, or from a pinned run's pin, and the tool writes there from any directory
(b32l, 2026-10-21). Each test pins one rule, on the walker (a scripted design run) and on
the spine's worker bridge (`worker_node.execute`, which every spine agent node runs). The
guard watches tracked files and HEAD only; what a refusal does to the files is
`target_guard.CHECKOUT_POLICY`'s, and each policy is tested here.
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
from test_design_worktree import _answers, _design
from tests_support import FakeModel, git, git_repo, render

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

    def complete(self, prompt, *, system="", workdir=None):
        if self.first is not None:
            self.first()
            self.first = None
        answer = super().complete(prompt, system=system, workdir=workdir)
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
                        lambda workdir, paths: watched.append(list(paths)) or real(workdir, paths))
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
                        lambda workdir, paths: watched.append(list(paths)) or real(workdir, paths))

    class Scratch(FakeModel):
        def complete(self, prompt, *, system="", workdir=None):
            if workdir is not None:
                # An agent's own notes in its run directory are its to write.
                (Path(workdir) / "scratch-notes.txt").write_text(prompt[:40])
            return super().complete(prompt, system=system, workdir=workdir)

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

    def complete(self, prompt, *, system="", workdir=None):
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
                        lambda workdir, paths: recorded.append(list(paths)) or real(workdir, paths))

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
