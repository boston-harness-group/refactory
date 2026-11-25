"""``scripts/test-changed``: the tests a local diff touches, run on eight workers.

The command exists so a person iterating outside a design run runs what the self-check
would run for the same diff, instead of the whole suite on every edit. These tests hold
that: its selection is `test_selection.select`'s (not a second selector), a change every
test reads selects the suite, the default is eight workers, a diff that reaches no test
runs nothing, and the three things the import graph cannot see — a fixture, a document, a
script — still select the test that reads them.
"""

from __future__ import annotations

import ast
import json
import os
import shlex
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
    structural test, a document, and the suite's own directory."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    tree = {
        "pyproject.toml": "",
        "refactory/pyproject.toml": "",
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


def _module() -> types.ModuleType:
    """The script itself, for the parts a run cannot show (its interpreter and environment)."""
    module = types.ModuleType("test_changed_script")
    module.__dict__["__file__"] = str(SCRIPT)
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), module.__dict__)  # noqa: S102
    return module


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


def test_a_typescript_spec_is_never_handed_to_pytest(tmp_path):
    """A spec is a test vitest runs: passing it to pytest fails the whole command before
    the Python tests run, and a spec alone must not look like Python coverage."""
    repo = _repo(tmp_path, {"packages/client/tests/contract.spec.ts": "it('x', () => {})\n"})
    (repo / "packages/client/tests/contract.spec.ts").write_text("it('y', () => {})\n")
    alone = _json(repo)
    assert alone["selection"] == [] and "pnpm test" in _run(repo).stdout

    (repo / "refactory/refactory_dsh/alpha.py").write_text("A = 2\n")
    mixed = _json(repo)
    assert "refactory/tests/test_alpha.py" in mixed["selection"]
    assert not [test for test in mixed["selection"] if test.endswith(".ts")]


def test_an_untracked_test_file_and_a_branch_commit_count(tmp_path):
    """A test the diff only created is still a test the diff touched; so is one a commit
    on the branch added."""
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "tests" / "test_new.py").write_text("def test_new():\n    pass\n")
    assert "refactory/tests/test_new.py" in _json(repo)["selection"]

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add a test")
    assert "refactory/tests/test_new.py" in _json(repo, "--base", "HEAD~1")["selection"]


def test_a_non_ascii_path_is_not_quoted_away(tmp_path):
    """`git diff` quotes a non-ASCII path unless it is asked for NUL-separated output; a
    quoted path matches no test and is silently dropped."""
    repo = _repo(tmp_path, {})
    path = repo / "refactory" / "tests" / "test_ünïcode.py"
    path.write_text("from refactory_dsh.alpha import A\n\n\ndef test_ü():\n    assert A == 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a Unicode test")
    path.write_text("from refactory_dsh.alpha import A\n\n\ndef test_ü():\n    assert A == 2\n")
    assert "refactory/tests/test_ünïcode.py" in _json(repo)["selection"]


def test_a_renamed_module_still_reaches_its_importers(tmp_path):
    """Renames list both paths (`--no-renames`), so the old name is in the diff and the
    test that imports it runs: it now fails at import."""
    repo = _repo(tmp_path, {})
    _git(repo, "mv", "refactory/refactory_dsh/alpha.py", "refactory/refactory_dsh/beta.py")
    record = _json(repo)
    assert "refactory/refactory_dsh/alpha.py" in record["touched"]
    assert "refactory/tests/test_alpha.py" in record["selection"]


def test_a_suite_wide_file_selects_the_whole_suite(tmp_path):
    """A file every test reads reaches every test, and no import graph says so."""
    for name in ("refactory/tests/conftest.py", "refactory/tests/model_credential.py"):
        repo = _repo(tmp_path / name.replace("/", "_"), {})
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((path.read_text() if path.exists() else "") + "\n# changed\n")
        record = _json(repo)
        assert record["selection"] == [SUITE] and record["suite_wide"] is True


def test_a_changed_fixture_selects_the_tests_that_name_it(tmp_path):
    """A fixture is data: read by path, imported by nothing."""
    repo = _repo(tmp_path, {
        "refactory/tests/fixtures/self_checks.json": "{}\n",
        "refactory/tests/test_reads_fixture.py":
            "from pathlib import Path\n\nFIXTURE = Path(__file__).parent / 'fixtures/self_checks.json'\n\n\n"
            "def test_reads_fixture():\n    assert FIXTURE.exists()\n",
    })
    (repo / "refactory/tests/fixtures/self_checks.json").write_text('{"a": 1}\n')
    selected = _json(repo)["selection"]
    assert selected == ["refactory/tests/test_reads_fixture.py"], selected


def test_a_document_and_a_script_select_the_test_that_owns_them(tmp_path):
    """The document census reads every document; a script's test is named for it."""
    repo = _repo(tmp_path, {
        "refactory/tests/test_docs_english_only.py": "def test_docs():\n    pass\n",
        "scripts/thing.py": "print('x')\n",
        "refactory/tests/test_thing.py":
            "from refactory_dsh.alpha import A\n\n\ndef test_thing():\n    assert A == 1\n",
    })
    (repo / "docs/guide.md").write_text("# Revised\n")
    assert _json(repo)["selection"] == ["refactory/tests/test_docs_english_only.py"]

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "doc")
    (repo / "scripts/thing.py").write_text("print('y')\n")
    selected = _json(repo)["selection"]
    assert "refactory/tests/test_thing.py" in selected
    assert "refactory/tests/test_reads_fixture.py" not in selected


def test_list_prints_the_selection_and_runs_nothing(tmp_path):
    """`--list` is for a pre-commit look: it must not run the failing test beside it."""
    repo = _repo(tmp_path, {}, failing=True)
    (repo / "refactory/tests/test_alpha.py").write_text("def test_alpha():\n    assert True\n")
    done = _run(repo, "--list")
    assert done.returncode == 0 and "running:" not in done.stdout
    assert "refactory/tests/test_alpha.py" in done.stdout


def test_a_diff_that_reaches_no_test_runs_nothing(tmp_path):
    """Docs select their census, and nothing else means no pytest: the suite here fails."""
    repo = _repo(tmp_path, {}, failing=True)
    (repo / "docs" / "guide.md").write_text("# A guide, revised\n")
    done = _run(repo, "--list")
    assert done.returncode == 0 and "running:" not in done.stdout
    assert _json(repo)["selection"] == []
    ran = _run(repo)
    assert ran.returncode == 0 and "nothing selected" in ran.stdout


def test_the_default_is_eight_workers(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "refactory_dsh" / "alpha.py").write_text("A = 2\n")
    words = shlex.split(_json(repo)["command"])
    assert words[:5] == [sys.executable, "-m", "pytest", "-q", "-p"]
    assert words[6:8] == ["-n", "8"]
    assert shlex.split(_json(repo, "--jobs", "2")["command"])[6:8] == ["-n", "2"]


def test_the_checkouts_venv_is_the_interpreter_and_its_package_is_first(tmp_path):
    """A worktree with no `.venv` must not run another checkout's code: its own venv, when
    it has one, and its own `refactory_dsh` on `sys.path`/`PYTHONPATH`."""
    module = _module()
    repo = _repo(tmp_path, {})
    assert module._python(repo) == sys.executable
    assert module._product(repo) is None
    assert module._env(repo).get("PYTHONPATH") == os.environ.get("PYTHONPATH")

    venv = repo / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(sys.executable)
    assert module._python(repo) == str(venv)
    assert module._reexec_target(repo) == str(venv)
    os.environ["TEST_CHANGED_IN_VENV"] = "1"
    try:
        assert module._reexec_target(repo) is None
    finally:
        del os.environ["TEST_CHANGED_IN_VENV"]

    (repo / "refactory" / "refactory_dsh" / "test_selection.py").write_text("")
    assert module._product(repo) == str(repo / "refactory")
    parts = module._env(repo)["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(repo / "refactory")
    assert parts[1:] == ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])


def test_pytest_options_after_the_separator_reach_pytest(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "refactory_dsh" / "alpha.py").write_text("A = 2\n")
    assert shlex.split(_json(repo, "--", "-x", "-k", "alpha")["command"])[-2:] == ["-k", "alpha"]


def test_full_runs_the_whole_suite_whatever_the_diff(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "refactory_dsh" / "alpha.py").write_text("A = 2\n")
    record = _json(repo, "--full")
    assert record["selection"] == [SUITE] and record["suite_wide"] is True


def test_a_base_that_is_not_a_commit_is_an_error(tmp_path):
    repo = _repo(tmp_path, {})
    done = _run(repo, "--base", "nope")
    assert done.returncode != 0 and "not a commit" in done.stderr


def test_a_checkout_without_commits_is_an_error_not_a_green(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "refactory" / "tests").mkdir(parents=True)
    done = _run(repo)
    assert done.returncode != 0 and "no commits" in done.stderr


def test_the_merge_base_is_the_default_base(tmp_path):
    """The branch's own commits are in the diff; what main has since added is not, so the
    diff never shows main's files as changes of the branch's."""
    repo = _repo(tmp_path, {})
    _git(repo, "checkout", "-q", "-b", "work")
    (repo / "refactory" / "tests" / "test_new.py").write_text("def test_new():\n    pass\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "work")
    _git(repo, "checkout", "-q", "-b", "mainline", "HEAD~1")
    (repo / "refactory" / "tests" / "test_main_only.py").write_text("def test_main():\n    pass\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "main moved")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "checkout", "-q", "work")

    record = _json(repo)
    assert record["base"] == _git(repo, "rev-parse", "work~1")
    assert "refactory/tests/test_new.py" in record["selection"]
    assert "refactory/tests/test_main_only.py" not in record["touched"]


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
    module = _module()
    tree = ast.parse((TESTS / "conftest.py").read_text())
    plugins = [element.value for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "pytest_plugins"
                       for target in node.targets)
               for element in node.value.elts]
    assert plugins, "conftest.py names no pytest_plugins"
    for name in plugins:
        assert f"{SUITE}/{name}.py" in module.SUITE_WIDE, name
