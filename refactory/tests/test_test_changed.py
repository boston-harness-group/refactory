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
        "refactory/alpha.py": "A = 1\n",
        "refactory/tests/test_alpha.py": ("from alpha import A\n\n\n"
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
    # A clone has main's tip; the command diffs against the merge base with it.
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
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
    (repo / "refactory" / "alpha.py").write_text("A = 2\n")
    record = _json(repo)
    expected = test_selection.select(
        ["refactory/alpha.py"], repo,
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

    (repo / "refactory/alpha.py").write_text("A = 2\n")
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
    path.write_text("from alpha import A\n\n\ndef test_ü():\n    assert A == 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a Unicode test")
    path.write_text("from alpha import A\n\n\ndef test_ü():\n    assert A == 2\n")
    assert "refactory/tests/test_ünïcode.py" in _json(repo)["selection"]


def test_a_renamed_module_still_reaches_its_importers(tmp_path):
    """Renames list both paths (`--no-renames`), so the old name is in the diff and the
    test that imports it runs: it now fails at import."""
    repo = _repo(tmp_path, {})
    _git(repo, "mv", "refactory/alpha.py", "refactory/beta.py")
    record = _json(repo)
    assert "refactory/alpha.py" in record["touched"]
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


def test_a_corpus_fixture_selects_the_helper_that_names_its_directory(tmp_path):
    """A corpus is read by a helper, not a top-level test, and named by its directory."""
    repo = _repo(tmp_path, {
        "refactory/tests/fixtures/corpus/one.json": "{}",
        "refactory/tests/corpus_support.py":
            "from pathlib import Path\n\nCORPUS = Path(__file__).parent / 'fixtures' / 'corpus'\n",
        "refactory/tests/test_corpus.py":
            "from corpus_support import CORPUS\n\n\ndef test_corpus():\n    assert CORPUS\n",
    })
    (repo / "refactory/tests/fixtures/corpus/one.json").write_text('{"b": 2}')
    assert _json(repo)["selection"] == ["refactory/tests/test_corpus.py"]


def test_a_document_and_a_script_select_the_test_that_owns_them(tmp_path):
    """The document census reads every document; a script's test is named for it."""
    repo = _repo(tmp_path, {
        "refactory/tests/test_ship_ready.py": "def test_ship():\n    pass\n",
        "refactory/tests/test_docs_api.py": "def test_api():\n    pass\n",
        "refactory/tests/test_docs_tutorial.py": "def test_tutorial():\n    pass\n",
        "refactory/tests/test_docs_english_only.py": "def test_docs():\n    pass\n",
        "scripts/thing.py": "print('x')\n",
        "refactory/tests/test_thing.py":
            "from alpha import A\n\n\ndef test_thing():\n    assert A == 1\n",
    })
    (repo / "docs/guide.md").write_text("# Revised\n")
    assert _json(repo)["selection"] == [
        "refactory/tests/test_ship_ready.py", "refactory/tests/test_docs_api.py",
        "refactory/tests/test_docs_tutorial.py", "refactory/tests/test_docs_english_only.py"]

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "doc")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "scripts/thing.py").write_text("print('y')\n")
    selected = _json(repo)["selection"]
    assert "refactory/tests/test_thing.py" in selected
    assert "refactory/tests/test_docs_english_only.py" not in selected

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "script")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    tutorial = repo / "refactory" / "docs" / "tutorial.html"
    tutorial.parent.mkdir(parents=True, exist_ok=True)
    tutorial.write_text("<p>a tutorial</p>\n")
    assert "refactory/tests/test_ship_ready.py" in _json(repo)["selection"]


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
    (repo / "refactory" / "alpha.py").write_text("A = 2\n")
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

    package = repo / "refactory" / "refactory_dsh"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "test_selection.py").write_text("")
    assert module._product(repo) == str(repo / "refactory")
    parts = module._env(repo)["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(repo / "refactory")
    assert parts[1:] == ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])


def test_the_checkouts_own_selector_is_used_and_its_absence_is_an_error(tmp_path):
    """`select()` must be the checkout's own: a worktree whose selector was deleted must
    fail, never quietly use the editable install's."""
    repo = _repo(tmp_path, {
        "refactory/refactory_dsh/__init__.py": "",
        "refactory/refactory_dsh/test_selection.py":
            "def repository_files(root):\n"
            "    return []\n\n\n"
            "def select(touched, root, **kwargs):\n"
            "    class Selection:\n"
            "        tests = ['refactory/tests/test_marker.py']\n"
            "    return Selection()\n",
        "refactory/tests/test_marker.py": "def test_marker():\n    pass\n",
    })
    assert _json(repo)["selection"] == ["refactory/tests/test_marker.py"]
    (repo / "refactory/refactory_dsh/test_selection.py").unlink()
    done = _run(repo, "--list")
    assert done.returncode != 0 and "test_selection" in done.stderr

    # A package directory that cannot be imported is refused too: deleting __init__.py
    # must not fall back to another checkout.
    (repo / "refactory/refactory_dsh" / "__init__.py").unlink()
    done = _run(repo, "--list")
    assert done.returncode != 0 and "no __init__.py" in done.stderr, done.stderr


def test_a_root_file_the_suite_reads_selects_its_reader(tmp_path):
    """`.python-version` and a workflow are read by the pin test; a snapshot by the test
    that names it. None of them is Python, and none may look like no change."""
    repo = _repo(tmp_path, {
        ".python-version": "3.12\n",
        ".github/workflows/ci.yml": "python-version: '3.12'\n",
        "refactory/tests/test_python_pin.py":
            "from pathlib import Path\n\n\n"
            "def test_the_pin_matches_ci():\n"
            "    assert (Path(__file__).parents[2] / '.python-version').is_file()\n",
        "refactory/tests/snapshots/one.json": "{}",
        "refactory/tests/test_snapshot.py":
            "from pathlib import Path\n\nSNAPSHOTS = Path(__file__).parent / 'snapshots'\n\n\n"
            "def test_snapshot():\n    assert SNAPSHOTS\n",
    })
    (repo / ".python-version").write_text("3.13\n")
    assert _json(repo)["selection"] == ["refactory/tests/test_python_pin.py"]

    (repo / ".python-version").write_text("3.12\n")
    (repo / ".github/workflows/ci.yml").write_text("python-version: '3.13'\n")
    assert _json(repo)["selection"] == ["refactory/tests/test_python_pin.py"]

    (repo / ".github/workflows/ci.yml").write_text("python-version: '3.12'\n")
    (repo / "refactory/tests/snapshots/one.json").write_text('{"a": 1}')
    assert _json(repo)["selection"] == ["refactory/tests/test_snapshot.py"]


def test_a_fixture_named_only_by_a_helper_selects_the_tests_that_import_it(tmp_path):
    """A helper can name the data and a test only import the helper: the test that reads
    the data is the one that imports the namer, whether or not its own text names it."""
    repo = _repo(tmp_path, {
        "refactory/tests/fixtures/rawdata/one.bin": "x",
        "refactory/tests/loaders.py":
            "from pathlib import Path\n\nDATA = Path(__file__).parent / 'fixtures' / 'rawdata'\n",
        "refactory/tests/test_loader.py":
            "from loaders import DATA\n\n\ndef test_loader():\n    assert DATA\n",
    })
    (repo / "refactory/tests/fixtures/rawdata/one.bin").write_text("y")
    assert _json(repo)["selection"] == ["refactory/tests/test_loader.py"]


def test_a_document_fixture_and_a_python_fixture_select_their_readers(tmp_path):
    """A fixture can be a document or a module; the data rule has to catch it before the
    document and `.py` exclusions do."""
    repo = _repo(tmp_path, {
        "refactory/tests/fixtures/excerpt.md": "a design\n",
        "refactory/tests/fixtures/helpers.py": "X = 1\n",
        "refactory/tests/test_reads_excerpt.py":
            "from pathlib import Path\n\nEXCERPT = Path(__file__).parent / "
            "'fixtures/excerpt.md'\n\n\ndef test_reads_excerpt():\n    assert EXCERPT.exists()\n",
    })
    (repo / "refactory/tests/fixtures/excerpt.md").write_text("a design, revised\n")
    assert _json(repo)["selection"] == ["refactory/tests/test_reads_excerpt.py"]


def test_a_python_fixture_selects_the_test_that_names_it(tmp_path):
    repo = _repo(tmp_path, {
        "refactory/tests/fixtures/helpers.py": "X = 1\n",
        "refactory/tests/test_uses_helpers.py":
            "from pathlib import Path\n\nHELPERS = Path(__file__).parent / "
            "'fixtures/helpers.py'\n\n\ndef test_uses_helpers():\n    assert HELPERS.exists()\n",
    })
    (repo / "refactory/tests/fixtures/helpers.py").write_text("X = 2\n")
    assert _json(repo)["selection"] == ["refactory/tests/test_uses_helpers.py"]


def test_an_unimported_helper_beside_an_unrelated_mention_is_not_a_reader(tmp_path):
    """A helper nobody imports names the data, and an unrelated test happens to mention
    it: neither reads the data, so the suite runs."""
    module = _module()
    repo = _repo(tmp_path, {
        "refactory/tests/dead_helper.py": "# fixtures/dead/one.bin\n",
        "refactory/tests/fixtures/dead/one.bin": "z",
        "refactory/tests/test_other.py": "# a note about dead data\n",
    })
    assert module._data_readers(repo, ["refactory/tests/fixtures/dead/one.bin"]) == [SUITE]


def test_data_named_only_by_an_unimported_helper_still_selects_the_suite(tmp_path):
    """A helper nobody imports names the file but no test reads it through the helper: the
    data is still unaccounted for, so the suite runs."""
    module = _module()
    repo = _repo(tmp_path, {
        "refactory/tests/orphan.py": "# guesses fixtures/lonely.bin\n",
        "refactory/tests/fixtures/lonely.bin": "z",
    })
    assert module._data_readers(repo, ["refactory/tests/fixtures/lonely.bin"]) == [SUITE]


def test_a_data_file_no_text_names_selects_the_suite(tmp_path):
    """One unnamed file is enough: nothing can clear it short of the suite."""
    repo = _repo(tmp_path, {"refactory/tests/fixtures/mystery.bin": "x"})
    (repo / "refactory/tests/fixtures/mystery.bin").write_text("y")
    assert _json(repo)["selection"] == [SUITE]


def test_a_named_and_an_unnamed_data_file_together_select_the_suite(tmp_path):
    """Nothing can clear a file nothing names, so it is not called "no change". The suite
    replaces what was selected, rather than joining it: a directory plus files collects
    twice."""
    repo = _repo(tmp_path, {
        "refactory/tests/fixtures/mystery.bin": "x",
        "refactory/tests/fixtures/named.json": "{}",
        "refactory/tests/test_reads_named.py":
            "from pathlib import Path\n\nF = Path(__file__).parent / 'fixtures/named.json'\n\n\n"
            "def test_reads_named():\n    assert F.exists()\n",
        "refactory/tests/test_docs_english_only.py": "def test_docs():\n    pass\n",
        "refactory/tests/orphan.py": "# names fixtures/nameless.bin\n",
        "refactory/tests/fixtures/nameless.bin": "z",
        "refactory/tests/fixtures/unspoken.bin": "u",
    })
    (repo / "refactory/tests/fixtures/nameless.bin").write_text("w")
    (repo / "refactory/tests/fixtures/unspoken.bin").write_text("v")
    (repo / "refactory/tests/fixtures/named.json").write_text('{"a": 1}')
    (repo / "refactory/tests/fixtures/mystery.bin").write_text("y")
    (repo / "docs/guide.md").write_text("# Revised\n")
    record = _json(repo)
    assert record["selection"] == [SUITE] and record["suite_wide"] is True


def test_only_the_python_suite_is_this_commands(tmp_path):
    """A packages/ or v1/ source is not a Python test: it selects none, and says which
    command covers it."""
    repo = _repo(tmp_path, {
        "packages/client/src/a.ts": "export const a = 1\n",
        "v1/build_backend.py": "X = 1\n",
    })
    (repo / "packages/client/src/a.ts").write_text("export const a = 2\n")
    (repo / "v1/build_backend.py").write_text("X = 2\n")
    record = _json(repo)
    assert record["selection"] == [] and record["not_selected"]
    assert "pnpm test" in _run(repo).stdout


def test_a_bytecode_cache_is_never_the_diff(tmp_path):
    """A run's own imports are not the change: a .pyc would otherwise select the suite."""
    repo = _repo(tmp_path, {}, failing=True)
    cache = repo / "refactory" / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "test_stale.cpython-312.pyc").write_bytes(b"x")
    (repo / "docs" / "guide.md").write_text("# Revised\n")
    record = _json(repo)
    assert record["touched"] == ["docs/guide.md"] and record["selection"] == []


def test_a_spaced_checkout_path_is_quoted_and_re_exec_runs(tmp_path, monkeypatch, capsys):
    """The printed command must be parseable back (shlex), and `main` must enter the
    checkout's venv rather than trust the interpreter that ran the script."""
    module = _module()
    repo = _repo(tmp_path / "with space", {})
    (repo / "refactory" / "alpha.py").write_text("A = 2\n")
    venv = repo / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(sys.executable)

    calls = []
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    monkeypatch.chdir(repo)
    assert module.main(["--json", "--base", "HEAD"]) == 0
    assert calls and calls[0][0] == str(venv)

    printed = json.loads(capsys.readouterr().out)
    assert shlex.split(printed["command"])[0] == str(venv)


def test_pytest_options_after_the_separator_reach_pytest(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "alpha.py").write_text("A = 2\n")
    assert shlex.split(_json(repo, "--", "-x", "-k", "alpha")["command"])[-2:] == ["-k", "alpha"]


def test_full_runs_the_whole_suite_whatever_the_diff(tmp_path):
    repo = _repo(tmp_path, {})
    (repo / "refactory" / "alpha.py").write_text("A = 2\n")
    record = _json(repo, "--full")
    assert record["selection"] == [SUITE] and record["suite_wide"] is True


def test_a_base_that_is_not_a_commit_is_an_error(tmp_path):
    repo = _repo(tmp_path, {})
    done = _run(repo, "--base", "nope")
    assert done.returncode != 0 and "not a commit" in done.stderr


def test_no_origin_main_asks_for_a_base_instead_of_looking_clean(tmp_path):
    """Without origin/main, a branch's committed work is invisible to a diff against
    HEAD; the command refuses rather than call it "no change"."""
    repo = _repo(tmp_path, {})
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    done = _run(repo)
    assert done.returncode != 0 and "no origin/main" in done.stderr, done.stderr
    assert "Traceback" not in done.stderr, done.stderr


def test_a_checkout_without_commits_is_an_error_not_a_green(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "refactory" / "tests").mkdir(parents=True)
    assert _run(repo).returncode != 0
    assert _run(repo, "--base", "HEAD").returncode != 0


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
