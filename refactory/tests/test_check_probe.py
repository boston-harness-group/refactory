"""The check's probe at prepare: a check that may not run is said in the run's first minute.

o55-dm149c's check passed ``-n 4`` to a pytest without pytest-xdist. Nothing ran the check
until the build's self-check, 82 minutes in, which then blamed the targeted command. Now
prepare probes the check where it will run (`check_probe`) -- pytest collects, every other
program is looked up -- on the walker and on the spine alike. The probe is advisory: the
change may add what the base lacks, so what it finds is a warning in the run's log and on
the approval card, and the run goes on. A self-check that cannot run shows pytest's own
error.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import refactory_dsh
from refactory_dsh import check_probe, failfast, lowering, nn, worktree
from refactory_dsh.design import DesignMode
from refactory_dsh.steering import Ledger
from tests_support import FakeModel, git, git_repo

SOURCE = str(Path(refactory_dsh.__file__).resolve().parents[1])
PYTEST = f"{sys.executable} -m pytest -q"
#: An option no installed plugin provides, as ``-n`` was without pytest-xdist.
UNPARSABLE = f"{PYTEST} --no-such-option-xyz tests"
BRIEF = "Make app.py print two."


def _target(root: Path) -> Path:
    repo = git_repo(root)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text("def test_app():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "tests")
    return repo


# ── the probe's command, for any check shape ──────────────────────────────────

def test_a_pytest_step_collects_with_every_option_kept():
    probe = check_probe.probe_command(".venv/bin/python -m pytest -q -n 4 --deselect a.py::t a.py")
    assert probe == ".venv/bin/python -m pytest -q -n 4 --deselect a.py::t a.py --collect-only -q"


def test_any_other_step_looks_its_program_up_and_builtins_stay():
    probe = check_probe.probe_command("cd web && FOO=1 pnpm test | tail -3")
    steps = probe.split(" && ")
    assert steps[0] == "cd web"
    assert "command -v pnpm" in steps[1] and "exit 127" in steps[1]
    assert "command -v tail" in steps[2]
    assert "pnpm test" not in probe, "the probe runs no test"
    assert "command -v node" in check_probe.probe_command("env -i A=1 node --test t.js")


def test_a_script_the_change_may_add_is_not_looked_up_but_an_installed_one_is():
    assert check_probe.probe_command("./scripts/check.sh --fast") is None
    assert "command -v .venv/bin/ruff" in check_probe.probe_command(".venv/bin/ruff check .")
    assert "command -v /usr/bin/env" in check_probe.probe_command("/usr/bin/env true")


def test_a_comment_never_swallows_the_collect_flag():
    probe = check_probe.probe_command("pytest -q  # the whole suite && more")
    assert probe == "pytest -q --collect-only -q"
    assert check_probe.probe_command("pytest -q '#not-a-comment'") == (
        "pytest -q '#not-a-comment' --collect-only -q")


def test_an_activation_step_stays_so_the_probe_collects_with_its_pytest():
    probe = check_probe.probe_command(". env/bin/activate && pytest -n 4")
    assert probe == ". env/bin/activate && pytest -n 4 --collect-only -q"
    assert check_probe.probe_command("source env/bin/activate; pytest").startswith(
        "source env/bin/activate && ")


def test_a_check_with_an_alternative_is_not_probed():
    assert check_probe.probe_command("pytest -n 4 || pytest") is None
    assert check_probe.probe_command("pytest -q '||' x") is not None, "a quoted || is an argument"


@pytest.mark.parametrize(("step", "runs_pytest"), [
    ("pytest -q", True),
    ("py.test", True),
    (".venv/bin/python -m pytest -n 4", True),
    ("python3.12 -X dev -m pytest", True),
    ("uv run --frozen pytest -q", True),
    ("FOO=1 env -i pytest", True),
    ("cd tests/pytest", False),
    ("pip install pytest", False),
    ("python -m pip install pytest", False),
    ("make pytest", False),
])
def test_only_the_command_word_makes_a_step_pytest(step, runs_pytest):
    assert check_probe.is_pytest(step) is runs_pytest


def test_a_check_of_builtins_alone_has_no_probe():
    assert check_probe.probe_command("cd sub") is None


@pytest.mark.parametrize(("returncode", "output", "fault"), [
    (4, "ERROR: usage: pytest [options]\npytest: error: unrecognized arguments: -n", True),
    (4, "ERROR: file or directory not found: tests/test_new.py", False),
    (4, "ERROR: not found: tests/a.py::test_new", False),
    (4, "", True),
    (127, "refactory: check probe: not found: pnpm", True),
    (126, "permission denied", True),
    (2, "ImportError while importing test module", False),
    (5, "no tests ran", False),
    (1, "", False),
])
def test_only_a_check_that_cannot_run_whatever_the_change_does_is_a_fault(returncode, output, fault):
    assert check_probe.probe_fault(returncode, output) is fault


# ── the probe in the prepared base ────────────────────────────────────────────

def _ctx(tmp_path: Path) -> nn.RunContext:
    target = _target(tmp_path / "target")
    run = tmp_path / "run"
    run.mkdir()
    worktree.create(target, "HEAD", "abc123", run)
    return nn.RunContext(run)


@pytest.mark.parametrize(("check", "says"), [
    (UNPARSABLE, "unrecognized arguments: --no-such-option-xyz"),
    ("no-such-program-xyz --check", "`no-such-program-xyz` is not installed or not on the check's PATH"),
    (".venv/bin/mytool --check", "`.venv/bin/mytool` is not installed or not on the check's PATH"),
])
def test_a_check_that_may_not_run_is_warned_about_in_the_tools_own_words(tmp_path, monkeypatch, capsys,
                                                                          check, says):
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    warning = check_probe.probe_check(_ctx(tmp_path), check)
    assert warning.startswith("Check may not run at base: ") and says in warning
    assert warning in capsys.readouterr().err, "the run's log says it too"


@pytest.mark.parametrize("check", [
    f"{PYTEST} tests",
    f"{PYTEST} tests/test_the_change_adds.py",  # a test file the change will write
    f"{sys.executable} app.py",
])
def test_a_check_that_can_run_passes_the_probe(tmp_path, monkeypatch, check):
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    assert check_probe.probe_check(_ctx(tmp_path), check) == ""


def test_the_probe_never_runs_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    ctx = _ctx(tmp_path)
    marker = tmp_path / "ran.txt"
    assert check_probe.probe_check(ctx, f"touch {marker} && {PYTEST} tests") == ""
    assert not marker.exists()


def test_the_check_does_not_install_again_after_the_probe(tmp_path, monkeypatch):
    """The probe makes the check's own install, in the run's worktree, under its record."""
    from refactory_dsh import repo_change, sandbox

    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    ctx = _ctx(tmp_path)
    installs = []
    monkeypatch.setattr(worktree, "_installs", lambda found: [["true", "install-marker"]])
    real = sandbox.run_command
    monkeypatch.setattr(sandbox, "run_command", lambda command, **kw: (
        installs.append(command) if "install-marker" in command else None) or real(command, **kw))
    assert check_probe.probe_check(ctx, f"{PYTEST} tests") == ""
    assert len(installs) == 1
    found = worktree.recorded(ctx.workdir)
    repo_change.prepare_check_environment(found, {}, amend=True, commit_lockfiles=True, run=ctx.workdir,
                                          home=tmp_path / "check-home", timeout=60)
    assert len(installs) == 1, "the check reused the probe's install"
    assert worktree.recorded(ctx.workdir).head == found.base, "the probe leaves the worktree at its base"


def test_without_a_worktree_the_check_is_probed_in_an_empty_directory(tmp_path):
    ctx = nn.RunContext(tmp_path / "run")
    warning = check_probe.probe_check(ctx, UNPARSABLE)
    assert warning.startswith("Check may not run in the artifact's directory: "), "it has no base"
    assert "its probe exited 4 (pytest's usage error: it could not parse the command line)" in warning
    assert "unrecognized arguments: --no-such-option-xyz" in warning
    assert check_probe.probe_check(ctx, f"{PYTEST} tests/test_the_change_adds.py") == ""


def test_the_approval_card_shows_the_probes_warning_with_the_tools_words():
    from refactory_dsh.design_ops import design_card, design_review_status

    facts = {"design_review_attempts_used": 1, "design_review_findings": "",
             "check_probe_warning": "Check may not run at base: its probe exited 4 (pytest's usage "
                                    "error).\npytest: error: unrecognized arguments: -n"}
    status = design_review_status(facts).splitlines()
    assert "Check may not run at base: its probe exited 4 (pytest's usage error). (advisory)" in status
    assert "- pytest: error: unrecognized arguments: -n" in status
    assert "unrecognized arguments: -n" in design_card(facts)
    assert "Check may not run" not in design_review_status({**facts, "check_probe_warning": ""})


# ── both runtimes warn at prepare and go on ───────────────────────────────────

def test_the_walker_warns_at_prepare_and_goes_on_to_research(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    repo = _target(tmp_path / "repo")
    mode = DesignMode()
    state = mode.state_dict()
    state["knobs"]["pr"]["value"] = False
    mode.load_state_dict(state)
    model = FakeModel([])
    mode.run(tmp_path / "run", model=model,
             task={"name": "probe_tool", "brief": BRIEF, "check": UNPARSABLE, "target": str(repo)})
    assert "Check may not run at base" in capsys.readouterr().err
    assert model.prompts, "the run went on to research"


#: The review's repros, which the base cannot run but the change may make runnable, and the
#: o55-dm149c shape with a brief that adds pytest-xdist (``-p no:xdist`` stands for its absence).
SPINE_CHECKS = {
    "vitest": ("node_modules/.bin/vitest run", "`node_modules/.bin/vitest` is not installed"),
    "mytool": (".venv/bin/mytool --check", "`.venv/bin/mytool` is not installed"),
    "xdist": (f"{PYTEST} -p no:xdist -n 4 tests", "unrecognized arguments: -n"),
}


def _spine_prepare(tmp_path: Path, knobs: dict) -> tuple[subprocess.CompletedProcess, Path]:
    state = DesignMode().state_dict()
    for knob, value in {"name": "probe_tool", "brief": BRIEF, "pr": False, **knobs}.items():
        state["knobs"][knob]["value"] = value
    run = tmp_path / "run"
    run.mkdir()
    Ledger.seed(run, BRIEF)
    lowering.write_declarations(state, run)
    node = lowering.to_workflow(state).nodes["prepare"]
    python, *runner = node.command.split()
    assert (python, runner[:2]) == ("python", ["-m", "refactory_dsh.run"])
    env = {**os.environ, **lowering.knob_environment(state), "PYTHONPATH": SOURCE,
           "PYTHONDONTWRITEBYTECODE": "1", "REFACTORY_WORKDIR": str(run),
           "DSH_WORKFLOW_NODE": json.dumps({"id": "prepare", "type": type(node).__name__})}
    done = subprocess.run([sys.executable, *runner], env=env, cwd=run, capture_output=True,
                          text=True, timeout=600)
    return done, run


@pytest.mark.usefixtures("configured_model_home")
@pytest.mark.parametrize("shape", SPINE_CHECKS)
def test_the_spine_warns_in_the_node_process_and_goes_on(tmp_path, monkeypatch, shape):
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    check, says = SPINE_CHECKS[shape]
    done, run = _spine_prepare(tmp_path, {"check": check, "target": str(_target(tmp_path / "repo"))})
    assert done.returncode == 0, done.stderr[-3000:]
    warning = json.loads((run / "facts.json").read_text())["check_probe_warning"]
    assert warning.startswith("Check may not run at base: ") and says in warning
    assert "Check may not run at base" in done.stderr


@pytest.mark.usefixtures("configured_model_home")
def test_the_spine_goes_on_without_a_worktree(tmp_path, monkeypatch):
    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    done, run = _spine_prepare(tmp_path, {"check": "python -m pytest -p tests.plugin tests",
                                          "worktree": False})
    # Whether this probe warns depends on the pytest on PATH; either way the run goes on.
    assert done.returncode == 0, done.stderr[-3000:]
    warning = json.loads((run / "facts.json").read_text())["check_probe_warning"]
    assert warning == "" or warning.startswith("Check may not run in the artifact's directory: ")


# ── a self-check that ran nothing shows pytest's own error ────────────────────

class _Usage:
    ok, timed_out, prep_failure, returncode = False, False, None, 4
    stdout = ""
    stderr = ("ERROR: usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n"
              "pytest: error: unrecognized arguments: -n\n")


def _broken(check: str) -> str:
    with pytest.raises(RuntimeError) as raised:
        failfast.targeted_prepass({}, check, lambda command, timeout: _Usage(), 180,
                                  selection=["tests/test_app.py"])
    return str(raised.value)


def test_a_self_check_that_ran_nothing_shows_pytests_error_not_a_guess():
    message = _broken("pytest -q -n 4 tests")
    assert "unrecognized arguments: -n" in message
    assert "usage error" in message
    assert "were not carried" not in message, "-n 4 was carried, so nothing says otherwise"


def test_options_are_said_to_be_dropped_only_when_they_were():
    assert failfast.dropped_options("pytest -q -n 4 tests", "pytest -q -n 4 tests/a.py") == []
    assert failfast.dropped_options("pytest -q -p no:cacheprovider tests",
                                    "pytest -q tests/a.py") == ["-p"]
    message = failfast.broken_command_message("pytest -q -x tests", "pytest -q tests/a.py",
                                              _Usage.stderr)
    assert "options -x were not carried" in message
