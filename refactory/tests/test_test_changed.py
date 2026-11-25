"""``scripts/test-changed``: the tests a local diff touches, run on eight workers.

The command exists so a person iterating outside a design run runs what the self-check
would run for the same diff, instead of the whole suite on every edit. These tests hold
that: its selection is `test_selection.select`'s (not a second selector), a change every
test reads selects the suite, the default is eight workers, and a diff that reaches no
test runs nothing.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import types
from pathlib import Path

from refactory_dsh import test_selection

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent.parent
SCRIPT = ROOT / "scripts" / "test-changed"
SUITE = "refactory/tests"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _repo(tmp_path: Path, files: dict[str, str], *, failing: bool = False) -> Path:
    """A checkout the command can run in: a changed module with a test that imports it, a
    structural test, and the suite's own directory."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    tree = {
        "pyproject.toml": "",
        "refactory/refactory_dsh/__init__.py": "",
        "refactory/refactory_dsh/alpha.py": "A = 1\n",
        "refactory/tests/test_alpha.py": ("from refactory_dsh.alpha import A\n\n\n"
                                          "def test_alpha():\n    assert A == 1\n"),
        "refactory/tests/test_structural.py": "def test_structural():\n    pass\n",
        "docs/guide.md": "# A guide\n",
        **files,
    }
    if failing:
        tree["refactory/tests/test_broken.py"] = "def test_broken():\n    assert False\n"
    for name, text in tree.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "init", "-q")
    for key, value in (("user.email", "test@test.com"), ("user.name", "Test")):
        _git(repo, "config", key, value)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """The command as a person runs it: from the checkout, by path."""
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=repo,
                          capture_output=True, text=True, timeout=300)


def _json(repo: Path, *args: str) -> dict:
    done = _run(repo, "--json", *args)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_its_selection_is_design_modes_self_check_selection(tmp_path):
    """The command does not select for itself: for the same diff it names what
    `test_selection.select` names, which is the function the self-check calls."""
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "refactory_dsh" / "alpha.py").write_text("A = 2\n")
    record = _json(repo)
    expected = test_selection.select(
        ["refactory/refactory_dsh/alpha.py"], repo,
        check_command=f"{sys.executable} -m pytest {SUITE}",
        files=test_selection.repository_files(repo))
    assert record["selection"] == expected.tests
    assert set(record["selection"]) == {"refactory/tests/test_alpha.py",
                                        "refactory/tests/test_structural.py"}
    assert record["base"] == _git(repo, "rev-parse", "HEAD")


def test_an_untracked_test_file_and_a_branch_commit_count(tmp_path):
    """A test the diff only created is still a test the diff touched; so is one a commit
    on the branch added, which a diff against `origin/main` sees too."""
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "tests" / "test_new.py").write_text("def test_new():\n    pass\n")
    assert "refactory/tests/test_new.py" in _json(repo)["selection"]

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add a test")
    assert "refactory/tests/test_new.py" in _json(repo, "--base", "HEAD~1")["selection"]


def test_a_suite_wide_file_selects_the_whole_suite(tmp_path):
    """A file every test reads reaches every test, and no import graph says so."""
    for name in ("refactory/tests/conftest.py", "pyproject.toml"):
        repo = _repo(tmp_path / name.replace("/", "_"), {})
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((path.read_text() if path.exists() else "") + "\n# changed\n")
        record = _json(repo)
        assert record["selection"] == [SUITE] and record["suite_wide"] is True


def test_a_diff_that_reaches_no_test_runs_nothing(tmp_path):
    """Docs select nothing, and nothing means no pytest: the suite here would fail."""
    repo = _repo(tmp_path, {}, failing=True)
    (repo / "docs" / "guide.md").write_text("# A guide, revised\n")
    done = _run(repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "nothing selected" in done.stdout and "running:" not in done.stdout


def test_the_default_is_eight_workers_and_the_checkouts_interpreter(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "refactory_dsh" / "alpha.py").write_text("A = 2\n")
    record = _json(repo)
    assert record["command"].split()[:5] == [
        sys.executable, "-m", "pytest", "-q", "-p"]
    assert "-n 8" in record["command"]
    assert "-n 2" in _json(repo, "--jobs", "2")["command"]


def test_pytest_options_after_the_separator_reach_pytest(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "refactory_dsh" / "alpha.py").write_text("A = 2\n")
    record = _json(repo, "--", "-x", "-k", "alpha")
    assert record["command"].endswith("-k alpha")


def test_it_runs_the_selection_and_keeps_pytests_exit_code(tmp_path):
    """One run, on the selected file; a failure in it is the command's exit code."""
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "tests" / "test_alpha.py").write_text(
        "def test_alpha():\n    assert True\n")
    passed = _run(repo, "--jobs", "1")
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "running:" in passed.stdout and "test_alpha.py" in passed.stdout

    (repo / "refactory" / "tests" / "test_alpha.py").write_text(
        "def test_alpha():\n    assert False\n")
    failed = _run(repo, "--jobs", "1")
    assert failed.returncode == 1, failed.stdout
    assert "FAILED" in failed.stdout


def test_every_plugin_the_suite_conftest_names_is_suite_wide():
    """A plugin named by `pytest_plugins` shapes every test, and no test imports it: it
    belongs with `conftest.py` in the list that selects the whole suite."""
    module = types.ModuleType("test_changed_script")
    source = SCRIPT.read_text()
    module.__dict__["__file__"] = str(SCRIPT)
    exec(compile(source, str(SCRIPT), "exec"), module.__dict__)  # noqa: S102 (a script, not a .py)
    tree = ast.parse((TESTS / "conftest.py").read_text())
    plugins = [element.value for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "pytest_plugins"
                       for target in node.targets)
               for element in node.value.elts]
    assert plugins, "conftest.py names no pytest_plugins"
    for name in plugins:
        assert f"{SUITE}/{name}.py" in module.SUITE_WIDE, name
