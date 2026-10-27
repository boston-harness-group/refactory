"""Pytest configuration for the refactory-dsh test harness."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Ops declared by a mode are resolved by module name at run time, so the test ops
# have to be importable by name rather than only by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The credential probe and the hook that skips the marked tests live together in
# `model_credential`, and a module imported by a test is not a plugin: its hooks are
# never called unless it is registered. Registering it here is what makes the skip
# fire, and it is also why the probe can stay deferred -- pytest calls the hook only
# at collection, and only this conftest pulls the module in.
pytest_plugins = ("model_credential", "repo_guard")

# PROMPT_MUTATE and expandable-KNOB_MUTATE shell out to `claude` by default.
# For the fast, deterministic harness we point that at /bin/true (empty output),
# which makes `default_prompt_rewriter` fall back to its deterministic prompt
# variants instead of calling a live LLM. The real DSH inner loop test runs the
# actual CEO route, so this only affects the surrogate/unit surface.
os.environ.setdefault("FACTORY_CLAUDE_BIN", "/bin/true")


@pytest.fixture(autouse=True)
def _isolate_refactory_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the project registry, run worktrees and new projects out of the real home.

    A design run with no target creates a project in the projects folder and a
    worktree under refactory's home (#351); a test must never leave either behind.
    """
    home = tmp_path_factory.mktemp("refactory-home")
    monkeypatch.setenv("REFACTORY_HOME", str(home / "home"))
    monkeypatch.setenv("REFACTORY_PROJECTS_DIR", str(home / "projects"))
    # A design run that passes delivers a pull request, and may create a repository
    # (#350), so no test may reach the person's own gh login: gh starts logged out.
    monkeypatch.setenv("GH_CONFIG_DIR", str(home / "gh"))
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _checkout_guard_leaves_this_checkout_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkout guard (dm-164) never watches the checkout under test.

    Every scripted agent the suite dispatches imports refactory from it, so the guard would
    snapshot it twice per dispatch, and refuse a test for an edit made there while the suite
    runs. Through the environment, so node and worker processes the tests start inherit it;
    the guard's own tests point it at temporary checkouts instead.
    """
    from refactory_dsh import target_guard
    from refactory_dsh.paths import REPO_ROOT

    monkeypatch.setenv(target_guard.UNWATCHED_ENV, str(REPO_ROOT))


# ── the sealed suite: no test reaches the person's DSH home or the real gh ──────
#
# A test resolved the person's own DSH home and touched a profile there in the middle of
# a live run's check (dm-267), and tests sent `gh api user` to the real GitHub, whose
# unauthenticated rate limit the next delivery on the machine then hit (dm-271). Both are
# now sealed off for every test: DSH_HOME points at an empty directory of the test's own,
# and the real home and the real gh are watched while the test runs.
#
# Tests that need the machine's own model credential (`needs_model_credential`) or a real
# DSH run (`integration`) keep the real DSH home: that is what they are marked for.

#: The DSH homes this session started with: the person's, before any test moved it.
_REAL_DSH_HOMES = tuple(sorted({
    form(os.path.expanduser(home))
    for home in (os.environ.get("DSH_HOME", "").strip(),
                 os.environ.get("REFACTORY_MODEL_DSH_HOME", "").strip(),
                 str(Path.home() / ".dsh"))
    if home
    for form in (os.path.abspath, os.path.realpath)
}))
_REAL_DSH_HOME_IN_TEXT = re.compile("|".join(re.escape(home) for home in _REAL_DSH_HOMES) + r"(?![\w.-])")
#: The real gh, when this machine has one.
_REAL_GH = shutil.which("gh")
_REAL_GH = os.path.realpath(_REAL_GH) if _REAL_GH else None

#: Events that name a path in their first argument, or in their first two.
_ONE_PATH = frozenset({
    "open", "os.listdir", "os.scandir", "os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown",
    "os.utime", "os.truncate", "shutil.rmtree", "glob.glob", "glob.glob/2",
})
_TWO_PATHS = frozenset({
    "os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "shutil.copytree", "shutil.move",
})
_SPAWN = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"})

#: Whether the current test is sealed, and what it reached while it was.
_SEAL: dict[str, Any] = {"home": False, "gh": False, "touched": []}


class SealBroken(PermissionError):
    """A test reached the person's DSH home or the real gh; the call is refused and the test fails."""


def _as_text(value: Any) -> str | None:
    if isinstance(value, int) or value is None:
        return None
    try:
        return os.fsdecode(os.fspath(value))
    except TypeError:
        return value if isinstance(value, str) else None


def _in_real_home(value: Any) -> bool:
    text = _as_text(value)
    if text is None:
        return False
    path = os.path.abspath(text)
    return any(path == home or path.startswith(home + os.sep) for home in _REAL_DSH_HOMES)


def _spawned(event: str, args: tuple[Any, ...]) -> tuple[list[Any], Any, dict[str, str] | None]:
    """The argv, working directory and environment a spawn event starts a child with."""
    if event == "subprocess.Popen":
        executable, argv, cwd, env = args
        argv = [argv] if isinstance(argv, (str, bytes, os.PathLike)) else list(argv or [])
        first = _as_text(argv[0]) if argv else None
        return ([executable] if executable and _as_text(executable) != first else []) + argv, cwd, env
    if event == "os.system":
        return [args[0]], None, None
    if event == "os.spawn":
        return [args[1], *list(args[2] or [])], None, args[3]
    return [args[0], *list(args[1] or [])], None, args[2]


def _runs_real_gh(argv: list[Any], env: dict[str, str] | None) -> bool:
    first = _as_text(argv[0]) if argv else None
    if not first or _REAL_GH is None:
        return False
    if os.sep not in first:
        found = shutil.which(first, path=(env if env is not None else os.environ).get("PATH"))
        first = found or first
    return os.path.realpath(first) == _REAL_GH


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not (_SEAL["home"] or _SEAL["gh"]) or (event not in _ONE_PATH and event not in _TWO_PATHS
                                              and event not in _SPAWN) or _SEAL.get("checking"):
        return
    _SEAL["checking"] = True
    try:
        _check(event, args)
    finally:
        _SEAL["checking"] = False


def _check(event: str, args: tuple[Any, ...]) -> None:
    if _SEAL["home"] and event in _ONE_PATH and args and _in_real_home(args[0]):
        reached = f"{event} {_as_text(args[0])}"
    elif _SEAL["home"] and event in _TWO_PATHS and (_in_real_home(args[0]) or _in_real_home(args[1])):
        reached = f"{event} {_as_text(args[0])} -> {_as_text(args[1])}"
    elif event in _SPAWN:
        argv, cwd, env = _spawned(event, args)
        if _SEAL["gh"] and _runs_real_gh(argv, env):
            reached = f"the real gh: {' '.join(str(_as_text(part)) for part in argv)}"
        elif _SEAL["home"] and (
            _in_real_home(cwd)
            or any(_REAL_DSH_HOME_IN_TEXT.search(_as_text(part) or "") for part in argv)
            or any(_REAL_DSH_HOME_IN_TEXT.search(value) for name, value in
                   (env if env is not None else os.environ).items()
                   if "DSH" in name or name.startswith("REFACTORY"))
        ):
            reached = f"{event} with the real DSH home: {' '.join(str(_as_text(part)) for part in argv)}"
        else:
            return
    else:
        return
    _SEAL["touched"].append(reached)
    raise SealBroken(f"refactory tests are sealed: {reached}")


sys.addaudithook(_audit)


@pytest.fixture(scope="session")
def _real_gh_shim(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A `gh` first on PATH that refuses, and logs, every call that reaches it.

    The fake (`tests/fake_gh.py`, through the tests' own `bin` directories) is put
    before it on PATH, so only a call that would have reached the real gh lands here.
    """
    bin_dir = tmp_path_factory.mktemp("sealed-gh")
    log = bin_dir / "calls.log"
    shim = bin_dir / "gh"
    shim.write_text(
        "#!/bin/sh\n"
        f"echo \"gh $*\" >> {shlex.quote(str(log))}\n"
        "echo 'refactory tests: the real gh is sealed off; use the fake (tests/fake_gh.py)' >&2\n"
        "exit 1\n"
    )
    shim.chmod(0o755)
    return bin_dir, log


@pytest.fixture(autouse=True)
def _sealed(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    _real_gh_shim: tuple[Path, Path],
) -> Iterator[None]:
    """Give the test an empty DSH home of its own, and fail it if it reaches the real one or the real gh."""
    real_home = any(request.node.get_closest_marker(name) is not None
                    for name in ("needs_model_credential", "integration"))
    if not real_home:
        home = tmp_path_factory.mktemp("dsh-home")
        monkeypatch.setenv("DSH_HOME", str(home))
        monkeypatch.setenv("REFACTORY_MODEL_DSH_HOME", str(home))
        # Every run needs a model now, and refuses without one (dm-258). The sealed home
        # holds the choice a person who picked one in Settings → Models has, so a test
        # that starts a run starts it; a test of the refusal or of the first-run choice
        # asks for `no_model_choice`.
        from tests_support import store_model_choice

        store_model_choice(home)
    bin_dir, log = _real_gh_shim
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    _SEAL.update(home=not real_home, gh=request.node.get_closest_marker("integration") is None, touched=[],
                 gh_log=log, gh_logged=log.stat().st_size if log.exists() else 0)
    try:
        yield
    finally:
        _SEAL.update(home=False, gh=False)
    touched = list(_SEAL["touched"])
    logged = _SEAL["gh_logged"]
    if log.exists() and log.stat().st_size > logged:
        with log.open() as calls:
            calls.seek(logged)
            touched += [f"the real gh: {line.strip()}" for line in calls if line.strip()]
    if touched:
        pytest.fail("this test reached outside its sealed environment (dm-267, dm-271):\n  "
                    + "\n  ".join(dict.fromkeys(touched)), pytrace=False)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run slow tests that spawn a real DSH CEO run",
    )
    parser.addoption(
        "--toolchain-gate",
        action="store_true",
        default=False,
        help=(
            "fail if a test marked `toolchain` did not run against its real "
            "external tool, and report what was verified"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: slow tests that spawn a real DSH CEO run"
    )
    config.addinivalue_line(
        "markers",
        "toolchain(name): needs a real external tool (lean, mathlib, sympy); "
        "checked by --toolchain-gate",
    )
    config.addinivalue_line(
        "markers",
        "needs_model_credential: needs a model credential the authoring sandbox "
        "accepts; the tests/model_credential.py hook skips it with a stated reason "
        "when this machine has none",
    )


#: nodeid -> tool, for every test that declares it needs a real external tool.
#:
#: A skip is invisible: the suite goes green and says nothing about the path the
#: skipped test covered, which is how two real Lean/mathlib defects hid. The
#: marker is the one place the set is written down, and the gate below reads it,
#: so it is not a list that drifts from the tests it is about (issue #44).
_TOOLCHAIN: dict[str, str] = {}
#: nodeid -> ("passed" | "failed" | "skipped", reason)
_TOOLCHAIN_RESULT: dict[str, tuple[str, str]] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        marker = item.get_closest_marker("toolchain")
        if marker is not None:
            _TOOLCHAIN[item.nodeid] = str(marker.args[0]) if marker.args else "unknown"
    if config.getoption("--run-integration", default=False):
        return
    skip = pytest.mark.skip(reason="pass --run-integration to run DSH inner-loop tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_deselected(items: list[pytest.Item]) -> None:
    """A deselected toolchain test did not run; `-k` must not hide it from the gate."""
    for item in items:
        marker = item.get_closest_marker("toolchain")
        if marker is not None:
            _TOOLCHAIN[item.nodeid] = str(marker.args[0]) if marker.args else "unknown"
            _TOOLCHAIN_RESULT[item.nodeid] = ("deselected", "")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record what happened to each marked test, including the reason it skipped."""
    if report.nodeid not in _TOOLCHAIN:
        return
    if report.skipped:
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = str(report.longrepr[2])
        _TOOLCHAIN_RESULT[report.nodeid] = ("skipped", reason)
    elif report.failed:
        _TOOLCHAIN_RESULT[report.nodeid] = ("failed", "")
    elif report.when == "call" and report.passed:
        _TOOLCHAIN_RESULT[report.nodeid] = ("passed", "")


def _unverified() -> list[tuple[str, str, str, str]]:
    """Every marked test that did not pass, with its tool, outcome, and reason."""
    problems: list[tuple[str, str, str, str]] = []
    for nodeid, tool in sorted(_TOOLCHAIN.items()):
        outcome, reason = _TOOLCHAIN_RESULT.get(nodeid, ("not run", ""))
        if outcome not in ("passed", "failed"):
            problems.append((nodeid, tool, outcome, reason))
    return problems


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """The gate: a green suite that skipped a tool-dependent test is not a pass."""
    if not session.config.getoption("--toolchain-gate", default=False):
        return
    if _unverified():
        session.exitstatus = 1


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    """Say which tool-dependent tests ran and which did not, whenever any are marked."""
    if not _TOOLCHAIN:
        return
    gated = config.getoption("--toolchain-gate", default=False)
    verified: dict[str, int] = {}
    for nodeid, tool in _TOOLCHAIN.items():
        if _TOOLCHAIN_RESULT.get(nodeid, ("", ""))[0] == "passed":
            verified[tool] = verified.get(tool, 0) + 1
    problems = _unverified()
    if not gated and not problems:
        return
    terminalreporter.write_sep("=", "toolchain gate")
    if verified:
        summary = ", ".join(f"{tool}: {count}" for tool, count in sorted(verified.items()))
        terminalreporter.write_line(f"verified against real tools: {summary}")
    if problems:
        terminalreporter.write_line(
            "UNVERIFIED: these tests need a real external tool and did not run:"
        )
        for nodeid, tool, outcome, reason in problems:
            terminalreporter.write_line(
                f"  [{tool}] {nodeid}: {outcome} -- {reason or 'no reason given'}"
            )
        terminalreporter.write_line(
            "a green suite here says nothing about the checker path; install the "
            "tools or accept that this machine did not verify it"
        )


# ── what a test that needs one of the sealed things is given instead ─────────────


@pytest.fixture
def fake_gh(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`gh` as `tests/fake_gh.py`, logged out, over an empty GitHub (dm-271).

    What a test that plans a delivery reaches instead of the real GitHub: a design run
    asks gh who is logged in before it plans its pull request. Write a `login` file in
    the returned root to log it in.
    """
    from tests_support import stub

    base = tmp_path_factory.mktemp("fake-gh")
    root = base / "github"
    root.mkdir()
    stub(base / "bin", "gh", f'exec "{sys.executable}" "{Path(__file__).with_name("fake_gh.py")}" "$@"')
    monkeypatch.setenv("PATH", f"{base / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_ROOT", str(root))
    return root


@pytest.fixture
def no_model_choice() -> Path:
    """The test's DSH home with no model chosen: a first run on a fresh install (dm-258)."""
    from refactory_dsh import dsh_profile

    home = Path(os.environ["DSH_HOME"])
    assert os.path.realpath(home) not in _REAL_DSH_HOMES, "no_model_choice needs the sealed DSH home"
    dsh_profile.model_choice_path(home).unlink(missing_ok=True)
    return home


@pytest.fixture
def graph_profile_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    """The test's DSH home with the `graph` profile install.sh leaves, linked to this checkout.

    What a test that starts the graph runtime needs, instead of the person's own profile
    (which, on a machine with several checkouts, is linked to another one).
    """
    from refactory_dsh.paths import REPO_ROOT
    from tests_support import graph_profile

    home = Path(os.environ["DSH_HOME"])
    assert os.path.realpath(home) not in _REAL_DSH_HOMES, "graph_profile_home needs the sealed DSH home"
    graph_profile(home, REPO_ROOT)
    return home


@pytest.fixture
def configured_model_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    """The test's DSH home with a headless route whose key is stored, as a placeholder.

    What a test that passes the authoring credential check needs (dm-257: the check now
    reads the store), instead of the person's own credentials.
    """
    from tests_support import model_home_with_a_key

    home = Path(os.environ["DSH_HOME"])
    assert os.path.realpath(home) not in _REAL_DSH_HOMES, "configured_model_home needs the sealed DSH home"
    return model_home_with_a_key(home)
