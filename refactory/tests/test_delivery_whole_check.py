"""Delivery runs the whole check once on a commit whose full check deselected the
self-check's passes (`delivery.whole_suite_owed`), so the check stays binding."""

from __future__ import annotations

from _delivery_support import *  # noqa: F403  (the helpers and fixtures these tests share)


def _tool(counter: Path, flag: Path) -> dict[str, str]:
    """A check that counts each run and fails once ``flag`` exists."""
    return {"tool.py": (f"import os\nopen({str(counter)!r}, 'a').write('x')\n"
                        f"raise SystemExit(1 if os.path.exists({str(flag)!r}) else 0)\n")}


def _owed_run(github: Path, tmp_path: Path, slug: str, *, fail: bool):
    repo = git_repo(tmp_path / "repo")
    _published(github, slug, repo)
    counter, before, flag = tmp_path / "check-runs", tmp_path / "before", tmp_path / "fail-flag"
    _child().setattr(delivery, "deliver_design", delivery_fakes.deliver_after, steps=[
        ["owe_whole_check", {"counter": str(counter), "before": str(before),
                             **({"fail_flag": str(flag)} if fail else {})}]])
    run = tmp_path / "run"
    result = _design(run, _tool(counter, flag), target=str(repo))
    return result, run, len(counter.read_text()) - int(before.read_text())


def test_delivery_runs_the_whole_check_once_when_main_did_not_move(github, tmp_path):
    """Main did not move, so there is no merge to check; the reviewed commit's check had
    deselected passes, so delivery runs the whole check on it once, and it passes."""
    result, run, runs = _owed_run(github, tmp_path, "acme/owed-check", fail=False)
    assert result.exit_ok, result.error
    assert runs == 1
    assert not delivery.whole_suite_owed(nn.RunContext(run, facts={"check": "python tool.py"}),
                                         Path(worktree.recorded(run).path),
                                         worktree.recorded(run).head)


def test_a_whole_check_that_fails_at_delivery_is_not_delivered_as_passing(github, tmp_path):
    """What the deselected check could not see fails the whole run at delivery: nothing is
    pushed as passing."""
    result, run, runs = _owed_run(github, tmp_path, "acme/owed-fails", fail=True)
    assert runs == 1
    said = json.dumps(_record(run)) + str(result.facts.get("delivery_result"))
    assert "the check failed on the merged result" in said, said
    assert not _remote_head(github, "acme/owed-fails",
                            json.loads((run / "design.json").read_text())["branch"])
