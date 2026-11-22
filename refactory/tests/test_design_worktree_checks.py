"""Design runs in a worktree: the check, setup, generate and install commands, and which failures are excused."""

from __future__ import annotations

from _design_worktree_support import *  # noqa: F403  (the helpers and fixtures these tests share)

pytestmark = pytest.mark.usefixtures("fake_gh")


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


def test_without_a_worktree_the_full_check_writes_no_guardedness_placeholder(repo, tmp_path, monkeypatch):
    """Only a worktree run can run the new tests on the base, so only it writes
    test-guardedness.json; the whole check (`design_ops._full_check`) writes no stand-in."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(repo))
    model = FakeModel(_answers(render({"tool.py": "print('built')\n"})))
    result = _design(tmp_path / "run", model, _mode(worktree=False))
    assert result.exit_ok, result.error
    # The full check ran (the run succeeded), verify it left no guardedness stand-in.
    assert (tmp_path / "run" / "check-record.json").exists(), "the whole check ran"
    assert not (tmp_path / "run" / "test-guardedness.json").exists(), \
        "test-guardedness.json should not be written on no-worktree runs"


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
    """A Node check that starts worker threads, as vitest does, failed under a 2 GB address-space cap (#325).

    RLIMIT_CPU derives from the backstop (not the inactivity limit) so a long-running
    active command is not killed by SIGXCPU (PIT-001).
    """
    import resource

    from refactory_dsh.sandbox import COMMAND_BACKSTOP_S

    tool = ("import json, resource\n"
            "print(json.dumps([resource.getrlimit(resource.RLIMIT_AS)[0], "
            "resource.getrlimit(resource.RLIMIT_CPU)[0]]))\n")
    result = _design(tmp_path / "run", FakeModel(_answers(render({"tool.py": tool}))), _mode(check_timeout_s=30),
                     target=str(repo))
    assert result.exit_ok, result.error
    evidence = json.loads((tmp_path / "run" / "authoring-evidence.json").read_text())
    desired_cpu = COMMAND_BACKSTOP_S * (os.cpu_count() or 1)
    hard = resource.getrlimit(resource.RLIMIT_CPU)[1]
    if hard != resource.RLIM_INFINITY and desired_cpu > hard:
        expected_cpu = hard
    else:
        expected_cpu = desired_cpu
    assert json.loads(evidence["validation"]["stdout"]) == [resource.getrlimit(resource.RLIMIT_AS)[0],
                                                            expected_cpu]


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
    (repo / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "-q", "-m", "manifest")
    monkeypatch.setenv("PATH", str(only_git))
    emit = render({"tool.py": "print('built')\n"})
    result = _design(tmp_path / "run", FakeModel(_answers(emit, emit)), target=str(repo))
    assert result.exit_ok is False
    assert "UNAVAILABLE: uv is not installed" in (result.error or "")
    found = worktree.recorded(tmp_path / "run")
    assert git(repo, "rev-list", "--count", f"{found.base}..{found.branch}") == "1", "no repair was attempted"


def test_a_setup_command_that_fails_fails_the_check_and_says_why(tmp_path):
    emit1 = render({"tool.py": "print('built')\n"})
    model = FakeModel(_answers(emit1))
    failed_setup = _design(tmp_path / "setup", model, _mode(setup="exit 3"),
                           target=str(git_repo(tmp_path / "plain")))
    # A setup that always fails is not change-caused → draft, not a builder round (D13).
    assert failed_setup.exit_ok is True
    draft = failed_setup.facts["draft_delivery"]
    assert draft["check_failing"] is True
    assert "code repair was not attempted" in draft["check_failure"]
    assert "exit 3" in failed_setup.facts.get("validation", "")
    # Exactly one emit — no builder round.
    emits = [p for p in model.prompts if "Write the artifact `" in p]
    assert len(emits) == 1


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
    model = FakeModel(_answers(emit1))
    failed_generate = _design(tmp_path / "run", model,
                              _mode(generate="exit 3"),
                              target=str(git_repo(tmp_path / "plain")))
    # A generate that always fails is not change-caused → draft, not a builder round (D13).
    assert failed_generate.exit_ok is True
    draft = failed_generate.facts["draft_delivery"]
    assert draft["check_failing"] is True
    assert "code repair was not attempted" in draft["check_failure"]
    assert "exit 3" in failed_generate.facts.get("validation", "")
    # Exactly one emit — no builder round.
    emits = [p for p in model.prompts if "Write the artifact `" in p]
    assert len(emits) == 1


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


def test_a_launcher_checkout_the_suite_leaves_unwatched_is_not_warned_about(
        repo, tmp_path, monkeypatch, capsys):
    """The test suite's own checkout (`REFACTORY_GUARD_UNWATCHED`) is not a launcher to warn
    about: a test on another worker writing there once read as this run's warning (CI)."""
    from refactory_dsh import target_guard

    launcher = git_repo(tmp_path / "launcher")
    monkeypatch.setenv("REFACTORY_PROJECT", str(launcher))
    monkeypatch.setenv(target_guard.UNWATCHED_ENV, str(launcher))
    result = _design(
        tmp_path / "run",
        _EditingModel(_answers(render({"tool.py": "print(1)\n"})),
                      lambda: launcher),
        target=str(repo))
    assert result.exit_ok is True, result.error
    warned = capsys.readouterr().err
    assert f"in {launcher.resolve()}" not in warned
    assert "files outside its worktree changed" not in warned


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
