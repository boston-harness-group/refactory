"""Target restore for worktree runs (dm-159): each test pins one core rule of the refusal and its lossless restore."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from refactory_dsh import agent_delivery, target_guard, worktree
from refactory_dsh import agent_delivery as ad
from test_design_worktree import _answers, _design
from tests_support import FakeModel, git, git_repo, render


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def _tree(root: Path) -> dict[str, tuple]:
    """Every entry under root except .git: (kind, bytes-or-link, mode)."""
    found = {}
    for directory, names, files in os.walk(root):
        names[:] = [n for n in names if n != ".git"]
        here = Path(directory)
        for name in files + [n for n in names if (here / n).is_symlink()]:
            path = here / name
            rel = str(path.relative_to(root))
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                found[rel] = ("link", os.readlink(path), None)
            else:
                found[rel] = ("file", path.read_bytes(), stat.S_IMODE(st.st_mode))
    return found


def _setup(tmp_path: Path, rid: str, *, run: Path | None = None, repo: Path | None = None):
    repo = repo or git_repo(tmp_path / "target")
    run = run or (tmp_path / "run")
    found = worktree.create(repo, "HEAD", rid, run)
    target_guard.record(run, repo, output_only=("emit",), also=(Path(found.path),))
    return repo, run, found


# ── s8 as the gate described it: qa_director starts WITH stray.py in its snapshot ──

def test_s8_parallel_reviewer_with_stray_in_its_snapshot(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0008a")
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    agent_delivery.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (repo / "stray.py").write_text("print('stray')\n")
    # qa_director starts while stray.py is in the target.
    agent_delivery.begin(run, "qa_director", "QA.", ["reviews/qa.md"])
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    with pytest.raises(ValueError, match="changed while agent 'code_reviewer'"):
        agent_delivery.finish(run, "code_reviewer")
    assert not (repo / "stray.py").exists()
    (run / "reviews" / "qa.md").write_text("VERDICT: pass\n")
    try:
        receipt = agent_delivery.finish(run, "qa_director")
        status = receipt["status"]
        error = ""
    except ValueError as exc:
        status = "refused"
        error = str(exc)
    present = (repo / "stray.py").exists()
    assert (status, present) == ("delivered", False), (status, present, error)


# ── s11: the person deleted a tracked file before the node; the agent recreates it ──

def test_s11_person_deleted_file_stays_deleted(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0011a")
    (repo / "app.py").unlink()
    before = _tree(repo)
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "app.py").write_text("print('agent recreated')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent") as refused:
        agent_delivery.finish(run, "emit")
    assert _tree(repo) == before
    assert "Could not restore" not in str(refused.value), str(refused.value)
    assert "could not be restored" not in str(refused.value).lower()


# ── s15: symlinks ──

def test_s15_untracked_symlink_retargeted(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0015a")
    os.symlink("app.py", repo / "link")
    before = _tree(repo)
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "link").unlink()
    (repo / "link").write_text("not a link any more\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent") as refused:
        agent_delivery.finish(run, "emit")
    assert _tree(repo) == before, str(refused.value)


def test_s15_clean_tracked_symlink_replaced(tmp_path):
    repo = git_repo(tmp_path / "target")
    os.symlink("app.py", repo / "link")
    git(repo, "add", "link")
    git(repo, "commit", "-q", "-m", "link")
    repo, run, _ = _setup(tmp_path, "probe0015b", repo=repo)
    before = _tree(repo)
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "link").unlink()
    (repo / "link").write_text("not a link any more\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent") as refused:
        agent_delivery.finish(run, "emit")
    status = git(repo, "status", "--porcelain")
    assert (_tree(repo) == before, status) == (True, ""), (str(refused.value), _tree(repo).get("link"), status)


# ── uncommitted mode change kept ──

def test_uncommitted_exec_mode_restored(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0010a")
    (repo / "tool.sh").write_text("#!/bin/sh\necho mine\n")
    os.chmod(repo / "tool.sh", 0o755)
    before = _tree(repo)
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "tool.sh").write_text("#!/bin/sh\necho agent\n")
    os.chmod(repo / "tool.sh", 0o644)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")
    assert _tree(repo) == before


# ── no false refusal: the run directory lies inside the target ──

def test_no_false_refusal_run_dir_inside_target(tmp_path):
    repo = git_repo(tmp_path / "target")
    run = repo / "runs" / "one"
    repo, run, found = _setup(tmp_path, "probe0004a", run=run, repo=repo)
    worktree.write_wrapper(found, run)
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    assert agent_delivery.finish(run, "emit")["status"] == "delivered"
    # A non-output-only agent writing its output in the run dir is delivered too.
    agent_delivery.begin(run, "verifier", "Verify.", ["verification.json"])
    (run / "verification.json").write_text("{}\n")
    assert agent_delivery.finish(run, "verifier")["status"] == "delivered"


# ── output-only refusal text: no contradictory "write to the worktree" ──

def test_output_only_refusal_text(tmp_path):
    repo, run, found = _setup(tmp_path, "probe0005a")
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "app.py").write_text("print('x')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError) as refused:
        agent_delivery.finish(run, "emit")
    text = str(refused.value)
    assert "Write to the worktree" not in text
    assert "only through the node's outputs" in text
    assert "revert it" not in text


# ── runs without a worktree are unchanged ──

def test_no_worktree_run_target_not_reverted(tmp_path):
    repo = git_repo(tmp_path / "target")
    (repo / "notes.txt").write_text("person\n")
    run = tmp_path / "run"
    target_guard.record(run, repo, output_only=("emit",))
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"])
    (repo / "app.py").write_text("print('person edited during the node')\n")
    (run / "emit.txt").write_text("Done.\n")
    with pytest.raises(ValueError, match="changed while agent") as refused:
        agent_delivery.finish(run, "emit")
    text = str(refused.value)
    left = (repo / "app.py").read_text()
    sets = sorted(str(p.relative_to(run)) for p in run.rglob("target-*") if p.is_dir())
    assert left == "print('person edited during the node')\n", (left, sets, text)
    assert "Nothing in the target was reverted" in text, text
    assert "reach the worktree" not in text, text


def test_no_worktree_non_output_only_not_watched(tmp_path):
    repo = git_repo(tmp_path / "target")
    run = tmp_path / "run"
    target_guard.record(run, repo, output_only=("emit",))
    agent_delivery.begin(run, "verifier", "Verify.", ["verification.json"])
    (repo / "scratch.txt").write_text("probe\n")
    (run / "verification.json").write_text("{}\n")
    assert agent_delivery.finish(run, "verifier")["status"] == "delivered"
    assert (repo / "scratch.txt").exists()


# ── a stray recorded AFTER this node's watch must not hide this node's own write ──

def test_stray_recorded_after_watch_does_not_hide_own_write(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0017b")
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    agent_delivery.begin(run, "qa_director", "QA.", ["reviews/qa.md"])
    agent_delivery.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (repo / "stray.py").write_text("print('code_reviewer stray')\n")
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    with pytest.raises(ValueError, match="changed while agent 'code_reviewer'"):
        agent_delivery.finish(run, "code_reviewer")
    assert not (repo / "stray.py").exists()
    # qa_director, watched before the stray was recorded, now writes the same path itself.
    (repo / "stray.py").write_text("print('qa_director own write')\n")
    (run / "reviews" / "qa.md").write_text("VERDICT: pass\n")
    try:
        status = agent_delivery.finish(run, "qa_director")["status"]
    except ValueError:
        status = "refused"
    assert (status, (repo / "stray.py").exists()) == ("refused", False)


# ── reused dispatch with an unchanged prompt: the re-watched snapshot's bytes are saved ──

def test_reused_dispatch_same_prompt_restores_persons_edit(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0012b")
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (repo / "app.py").write_text("print('dirty')\n")
    (run / "emit.txt").write_text("Done.\n")
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")
    (repo / "app.py").write_text("print('person kept this')\n")
    first = agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    (repo / "app.py").write_text("print('agent again')\n")
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent") as refused:
        agent_delivery.finish(run, "emit")
    assert (repo / "app.py").exists(), str(refused.value)
    assert (repo / "app.py").read_text() == "print('person kept this')\n", str(refused.value)


# ── the incident, end to end through a scripted design run (walker) ──

def test_incident_scripted_design_run(tmp_path):
    import json as _json
    import re
    import sys as _sys

    from refactory_dsh import agent_delivery as ad
    from test_design_worktree import _answers, _design
    from tests_support import FakeModel, render

    repo = git_repo(tmp_path / "repo")
    # A generator that, like gen_discovery, picks its output from where it lives.
    (repo / "gen.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('generated.json').write_text('{}\\n')\n")
    git(repo, "add", "gen.py")
    git(repo, "commit", "-q", "-m", "gen")
    # The person's own state: an uncommitted edit and an untracked file.
    (repo / "app.py").write_text("print('mine, uncommitted')\n")
    (repo / "notes.txt").write_text("untracked notes\n")
    before = _tree(repo)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    seen: dict[str, object] = {}

    class Incident(FakeModel):
        def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
            answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
            if "Write the artifact `" in prompt and "python" not in seen:
                block = prompt[prompt.index(ad.CONTRACT_START):prompt.index(ad.CONTRACT_END)]
                execution = _json.loads(block[block.index("{"):])
                seen["python"] = execution["python"]
                seen["prompt"] = prompt
                (repo / "app.py").write_text("print('agent edit')\n")
                (repo / "added.py").write_text("print('agent added')\n")
                subprocess.run([execution["python"], str(repo / "gen.py")], cwd=scratch, check=True)
                seen["generated"] = (repo / "generated.json").exists()
            return answer

    result = _design(tmp_path / "run", Incident(_answers(render({"tool.py": "print('built')\n"}))),
                     target=str(repo))
    assert seen.get("generated") is True, "the generator wrote into the target"
    assert result.exit_ok is False
    assert "changed while agent 'emit' ran" in (result.error or ""), result.error
    assert _tree(repo) == before, sorted(set(_tree(repo).items()) ^ set(before.items()))
    receipt = ad.receipts_in(tmp_path / "run")["emit"]
    assert receipt["status"] != "delivered", receipt["status"]
    prompt = str(seen["prompt"])
    found = worktree.recorded(tmp_path / "run")
    assert f"worktree at {found.path}" in prompt
    assert "nested dispatch only" in prompt
    assert str(Path(ad.__file__).resolve().parent) in prompt.split(ad.CONTRACT_END)[0]
    print("REFUSAL:", result.error)
    print("CONTRACT:", prompt[prompt.index(ad.CONTRACT_START):prompt.index("{", prompt.index(ad.CONTRACT_START))])


def test_s8_then_own_write_is_still_restored(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0008c")
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    agent_delivery.begin(run, "code_reviewer", "Review.", ["reviews/code.md"])
    (repo / "stray.py").write_text("print('stray')\n")
    agent_delivery.begin(run, "qa_director", "QA.", ["reviews/qa.md"])
    (run / "reviews" / "code.md").write_text("VERDICT: pass\n")
    with pytest.raises(ValueError, match="changed while agent 'code_reviewer'"):
        agent_delivery.finish(run, "code_reviewer")
    (repo / "stray.py").write_text("print('qa_director own write')\n")
    (run / "reviews" / "qa.md").write_text("VERDICT: pass\n")
    with pytest.raises(ValueError, match="changed while agent 'qa_director'"):
        agent_delivery.finish(run, "qa_director")
    assert not (repo / "stray.py").exists()


# ── CEO route: the same in-flight node is begun again with a changed prompt ──

def test_inflight_rebegin_keeps_persons_uncommitted_bytes(tmp_path):
    repo, run, _ = _setup(tmp_path, "probe0012c")
    (repo / "app.py").write_text("print('person, uncommitted')\n")
    agent_delivery.begin(run, "emit", "Write emit.txt.", ["emit.txt"], reuse=True)
    # The agent edits the person's dirty file while the node is in flight ...
    (repo / "app.py").write_text("print('agent, mid-node')\n")
    # ... and the bridge re-supplies the node with a changed prompt (a new ledger entry).
    agent_delivery.begin(run, "emit", "Write emit.txt. A correction arrived.", ["emit.txt"], reuse=True)
    (run / "emit.txt").write_text("===FILE: tool.py===\nprint(1)\n")
    with pytest.raises(ValueError, match="changed while agent"):
        agent_delivery.finish(run, "emit")
    assert (repo / "app.py").read_text() == "print('person, uncommitted')\n"


# ── A scripted design run whose verifier writes into the target ──

def _tree_with_modes(root: Path) -> dict:
    found = {}
    for directory, names, files in os.walk(root):
        names[:] = [n for n in names if n != ".git"]
        for name in files:
            path = Path(directory) / name
            st = os.lstat(path)
            found[str(path.relative_to(root))] = (path.read_bytes() if not stat.S_ISLNK(st.st_mode)
                                                  else os.readlink(path), stat.S_IMODE(st.st_mode))
    return found


def test_scripted_design_run_verifier_writes_into_target(tmp_path):
    repo = git_repo(tmp_path / "repo")
    (repo / "notes.txt").write_text("person's untracked notes\n")
    before = _tree_with_modes(repo)

    class Verifier(FakeModel):
        def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
            answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
            block = prompt[prompt.index(ad.CONTRACT_START):prompt.index(ad.CONTRACT_END)] if ad.CONTRACT_START in prompt else ""
            if block and json.loads(block[block.index("\n{") + 1:])["node"] == "verifier":
                (repo / "app.py").write_text("print('verifier edited the target')\n")
                (repo / "probe_out.txt").write_text("stray\n")
            return answer

    result = _design(tmp_path / "run", Verifier(_answers(render({"tool.py": "print('built')\n"}))),
                     target=str(repo))
    # The verifier is an optional review branch (#475): the guard still refuses its write and
    # restores the target, and its refused delivery is a failed node, not a failed run.
    assert result.exit_ok, result.error
    assert _tree_with_modes(repo) == before
    assert ad.receipts_in(tmp_path / "run")["verifier"]["status"] == "failed"
    events = [json.loads(line) for line in
              (tmp_path / "run" / ".refactory" / "events.jsonl").read_text().splitlines() if line.strip()]
    assert any(e.get("kind") == "node_failed" and e.get("node_id") == "verifier"
               and "changed while agent 'verifier' ran" in e.get("summary", "") for e in events)
