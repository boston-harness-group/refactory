"""Which tests design mode's self-check runs for an emission, in the order it runs them (b41a).

The engine reads the repository's own imports; it names no test or file of any
repository. Classified as a *product* module in ``data/boundary.json``, and no core
module imports it.

The selection, in order:

- **On a fix-up, the tests the last self-check found failing**, as ids.
- **Engine tests**, only when the emission changed a hub. A *hub* is a source module at
  least a quarter of the repository's Python test files import (`HUB_SHARE`): the
  engine or graph modules most of the suite is built on. The engine tests are the
  structural tests (test files that import no module of the repository: they read it
  as files, its layout, records and docs, so no import says a change cannot reach them)
  and the tests that import every hub, the ones built on the whole engine.
- **The test files the emission touched.**
- **The importers of a changed file the check does not run.** A deleted test file, a test
  another suite collects, or a product module named like a test (``test_selection.py``)
  is not run, and does not widen the selection to the structural tests, but a collected
  test that imports it now breaks: the file is resolved and its importers run.
- **The last check's failures** that are not inherited (dm-128's set), as test ids.
- **The tests that import a module the emission changed**, depth 1, and, when a source
  file changed but no hub did, the structural tests: cheapest first, so a slow file at
  the time limit does not hide cheap ones behind it (`estimated_costs`). A file a chunk
  was stopped with at the limit runs in a chunk of its own until one finishes it
  (`run_alone`); until then its time is a lower bound, which never lowers its estimate.

For a TypeScript runner only the spec files the emission touched are selected: the
check's own vitest invocation (``vitest run``) has no related-file selection, and
``--related`` there is an unknown option. A docs-only emission selects nothing.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: A source module at least this share of the Python test files import is a hub.
HUB_SHARE = 0.25

#: Files that are only read, never run or imported: an emission of these alone selects nothing.
DOC_SUFFIXES = (".md", ".rst", ".txt")

_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")


@dataclass
class Selection:
    """The ordered selection, and how it was made, for the self-check's record."""

    tests: list[str] = field(default_factory=list)
    engine: list[str] = field(default_factory=list)
    hubs_changed: list[str] = field(default_factory=list)
    touched_tests: list[str] = field(default_factory=list)
    last_failures: list[str] = field(default_factory=list)
    importers: list[str] = field(default_factory=list)
    structural: list[str] = field(default_factory=list)
    first: list[str] = field(default_factory=list)
    alone: list[str] = field(default_factory=list)  # files that run one per chunk (`run_alone`)

    def record(self) -> dict[str, Any]:
        return {"first": self.first, "engine": self.engine, "hubs_changed": self.hubs_changed,
                "touched_tests": self.touched_tests, "last_failures": self.last_failures,
                "importers": self.importers, "structural": self.structural,
                "alone": self.alone}


def is_test_file(path: str) -> bool:
    """Whether a runner collects ``path`` as a test module (pytest's or vitest's naming)."""
    from refactory_dsh.base_check import _TEST_FILE_PATTERNS

    return bool(_TEST_FILE_PATTERNS.search(path))


def repository_files(root: Path) -> list[str]:
    """The repository's files, from git; a directory that is not a checkout is walked."""
    try:
        listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
                                timeout=60, check=True).stdout.decode(errors="replace")
        return [name for name in listed.split("\0") if name]
    except (OSError, subprocess.SubprocessError):
        skip = {".git", ".venv", "node_modules", "__pycache__"}
        return [str(path.relative_to(root)) for path in root.rglob("*")
                if path.is_file() and not skip & set(path.relative_to(root).parts)]


def _import_roots(files: list[str]) -> list[Path]:
    """Where an absolute import resolves from: the root, each Python project, its ``src``."""
    projects = {Path(name).parent for name in files if Path(name).name in _PROJECT_MARKERS}
    roots = [Path("")]
    for project in sorted(projects, key=lambda p: (len(p.parts), str(p))):
        roots += [project, project / "src"]
    return roots


def _resolve(module: str, roots: list[Path], known: set[str]) -> str | None:
    parts = module.replace(".", "/")
    for base in roots:
        for candidate in (f"{parts}.py", f"{parts}/__init__.py"):
            name = (base / candidate).as_posix()
            if name in known:
                return name
    return None


def imports_of(test: str, text: str, roots: list[Path], known: set[str]) -> set[str]:
    """The repository files a Python test file imports, depth 1.

    ``from package import name`` resolves to the submodule when ``name`` is one, and to
    the package otherwise. A test's own directory comes first, as pytest puts it on the
    path; a relative import resolves from the test's package.
    """
    try:
        tree = ast.parse(text, filename=test)
    except (SyntaxError, ValueError):
        return set()
    own = [Path(test).parent, *roots]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve(alias.name, own, known)
                if resolved:
                    found.add(resolved)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = Path(test).parent
                for _ in range(node.level - 1):
                    package = package.parent
                bases, module = [package], node.module or ""
            else:
                bases, module = own, node.module or ""
            for alias in node.names:
                dotted = f"{module}.{alias.name}" if module else alias.name
                resolved = _resolve(dotted, bases, known) or (
                    _resolve(module, bases, known) if module else None)
                if resolved:
                    found.add(resolved)
    return found


def import_map(root: Path, files: list[str], extra: tuple[str, ...] = (),
               scope: list[str] | None = None) -> dict[str, set[str]]:
    """Each Python test file the check runs (in ``scope``) and the repository files it imports.

    ``extra`` names files that resolve though they are gone, the ones an emission
    deleted, so a test still importing one is found.
    """
    known = set(files) | set(extra)
    roots = _import_roots(files)
    found: dict[str, set[str]] = {}
    for name in files:
        if not name.endswith(".py") or not is_test_file(name) or not _in_scope(name, scope):
            continue
        try:
            text = (root / name).read_text(errors="replace")
        except OSError:
            continue
        found[name] = imports_of(name, text, roots, known) - {name}
    return found


def pytest_scope(check_command: str, root: Path) -> list[str] | None:
    """The paths the check's pytest steps collect from, or None when one collects everything.

    A path is a step's argument that names a file or directory of the repository; a step
    that names none collects from the root, and so does a check with no pytest step.
    """
    import shlex

    from refactory_dsh.base_check import _runner, _steps

    scope: list[str] = []
    steps = [step for step in _steps(check_command) if _runner(step) == "pytest"]
    if not steps:
        return None
    for step in steps:
        try:
            words = shlex.split(step)
        except ValueError:
            return None
        named = [word.rstrip("/") for word in words[1:]
                 if not word.startswith("-") and (root / word).exists()
                 and Path(word).name not in ("python", "python3", "pytest")]
        if not named:
            return None
        scope += named
    return scope


def _in_scope(path: str, scope: list[str] | None) -> bool:
    return scope is None or any(path == entry or path.startswith(entry + "/") for entry in scope)


def _source(path: str) -> bool:
    from refactory_dsh.changed_functions import is_test_path

    return not is_test_path(path)


def hubs(imports: dict[str, set[str]]) -> set[str]:
    """The source modules at least `HUB_SHARE` of the test files import."""
    if not imports:
        return set()
    counts: dict[str, int] = {}
    for modules in imports.values():
        for module in modules:
            counts[module] = counts.get(module, 0) + 1
    floor = HUB_SHARE * len(imports)
    return {module for module, count in counts.items() if count >= floor and _source(module)}


def _failures(prev_record: dict[str, Any], root: Path) -> list[str]:
    """The last check's failures that are not inherited, as ids, in the order it found them."""
    if prev_record.get("status") != "fail":
        return []
    inherited = set(prev_record.get("inherited", []))
    ids: list[str] = []
    for runner in prev_record.get("runners", []):
        for failure in runner.get("failures", []):
            test_id = str(failure.get("test_id") or "")
            if not test_id or test_id in inherited or test_id in ids:
                continue
            if (root / _file_of(test_id)).is_file() and runnable(test_id) not in ids:
                ids.append(runnable(test_id))
    return ids


def _file_of(test_id: str) -> str:
    return test_id.split("::")[0] if "::" in test_id else test_id.split(" > ")[0].strip()


def runnable(test_id: str) -> str:
    """The test id, or its file when the id cannot be passed back to pytest.

    `base_check.parse_test_failures` cuts a parametrized id at the first character it
    does not read, such as a space in a parameter (``test_x[a b]`` becomes ``test_x[a``);
    pytest finds no such test and stops the whole command. Its file runs instead.
    """
    if "::" in test_id and test_id.count("[") > test_id.count("]"):
        return _file_of(test_id)
    return test_id


def _whole_files(chunk: dict[str, Any]) -> set[str]:
    """The files a timed chunk ran whole, not only some of their tests by id."""
    return {_file_of(name) for name in chunk.get("tests", [])
            if "::" not in name and " > " not in name}


def _chunk_costs(chunk: dict[str, Any]) -> dict[str, tuple[float, bool]]:
    """Each file's seconds from one timed chunk, and whether they are only a lower bound.

    With ``"durations"`` (per-test seconds from ``--junitxml``) a file costs the sum of its
    tests'. Without them only a chunk of one file says what that file took: its elapsed
    time. A chunk of several files without durations says nothing about any one of them.
    A chunk stopped at the limit (``timed_out``) gives lower bounds, and so does a file
    the chunk ran only some tests of, by id: the whole file takes at least as long.
    """
    stopped = bool(chunk.get("timed_out"))
    whole = _whole_files(chunk)
    durations = chunk.get("durations")
    if durations:
        seconds: dict[str, float] = {}
        for duration in durations:
            name = _file_of(duration["test_id"])
            seconds[name] = seconds.get(name, 0.0) + float(duration["seconds"])
        return {name: (value, stopped or name not in whole) for name, value in seconds.items()}
    files = list(dict.fromkeys(_file_of(name) for name in chunk.get("tests", [])))
    if len(files) == 1:
        return {files[0]: (float(chunk.get("s", 0)), stopped or files[0] not in whole)}
    return {}


def _timed(entries: list[dict[str, Any]]) -> dict[str, tuple[float, bool]]:
    """Each test file's seconds from the chunks earlier self-checks timed (`_chunk_costs`),
    and whether that is still only a lower bound.

    A finished time beats a lower bound, and between two of a kind the higher stands, so a
    lower bound only ever raises what is known of a file, never lowers it.
    """
    timed: dict[str, tuple[float, bool]] = {}
    for entry in entries:
        for chunk in entry.get("timings", []):
            if not chunk.get("counted"):
                continue
            for name, (value, bound) in _chunk_costs(chunk).items():
                if name not in timed or (timed[name][1] and not bound):
                    timed[name] = (value, bound)
                elif timed[name][1] == bound:
                    timed[name] = (max(timed[name][0], value), bound)
    return timed


def costs_from_record(entries: list[dict[str, Any]], root: Path) -> dict[str, float]:
    """Each test file's seconds, from the chunks earlier self-checks timed (`_timed`).

    A chunk of several files without per-test durations leaves each file's estimate as it
    was; its files run one per chunk on the next self-check (`run_alone`).
    """
    return {name: value for name, (value, _) in _timed(entries).items()}


def lower_bounds(entries: list[dict[str, Any]]) -> list[str]:
    """The test files whose time is only a lower bound: no chunk has finished them whole."""
    return [name for name, (_, bound) in _timed(entries).items() if bound]


def run_alone(entries: list[dict[str, Any]]) -> list[str]:
    """The test files the next self-check runs one per chunk: each file of a chunk the limit
    stopped, until a later chunk finishes it whole.

    A stopped chunk writes no per-test durations (its runner is killed), so when it held
    several files its time says nothing about which of them was slow: a small slow file
    would keep its small estimate, run first, and cut everything behind it on every
    self-check (dm-247). Alone, each file gets a time or a lower bound of its own, and a
    file that has only a bound cannot take a chunk of other files down with it when the
    limit stops it again.
    """
    alone: dict[str, None] = {}
    for entry in entries:
        for chunk in entry.get("timings", []):
            if not chunk.get("counted"):
                continue
            if chunk.get("timed_out"):
                alone.update(dict.fromkeys(_file_of(name) for name in chunk.get("tests", [])))
                continue
            for name, (_, bound) in _chunk_costs(chunk).items():
                if not bound:
                    alone.pop(name, None)
    return list(alone)


def _size(root: Path, name: str) -> int:
    try:
        return max(1, (root / name).stat().st_size)
    except OSError:
        return 1


def estimated_costs(tests: list[str], root: Path, known: dict[str, float],
                    bounds: list[str] = ()) -> dict[str, float]:
    """A test file's cost: the seconds a self-check timed it at, else its size.

    Sizes are put in seconds at the rate the finished files ran, when there are any. A
    file of ``bounds`` has only a lower bound, which is no rate it ran at: it costs the
    higher of its bound and its size estimate, so a bound never lowers an estimate. A
    file stopped near the limit has a small bound; costed by that alone, it would run
    early the next time and cut everything behind it. With no finished file there is no
    rate, and sizes stay in bytes, which a bound in seconds cannot be weighed against:
    the files with only a bound then go last, in the order of their bounds.
    """
    lower = set(bounds)
    timed = [name for name in tests if name in known and name not in lower]
    rate = (sum(known[name] for name in timed) / sum(_size(root, name) for name in timed)
            if timed else 1.0)
    costs = {name: known.get(name, _size(root, name) * rate) for name in tests}
    for name in lower & costs.keys():
        costs[name] = max(costs[name], _size(root, name) * rate)
    if not timed:
        last = max((cost for name, cost in costs.items() if name not in lower), default=0.0)
        for name in lower & costs.keys():
            costs[name] = last + known.get(name, 0.0)
    return costs


def select(touched: list[str], root: Path, prev_record: dict[str, Any] | None = None, *,
           check_command: str = "", files: list[str] | None = None,
           imports: dict[str, set[str]] | None = None, first: list[str] = (),
           costs: dict[str, float] | None = None, bounds: list[str] = (),
           alone: list[str] = ()) -> Selection:
    """The self-check's selection for an emission that changed ``touched`` in ``root``.

    ``touched`` are paths from the repository root (`base_check._all_changed_files`);
    ``files`` the repository's files, listed from git when not given. Only Python tests
    the check's pytest steps collect are run (`pytest_scope`), and the hubs are counted
    among them. A touched file the check does not run — a deleted test, a test another
    suite collects, a product module named like a test — is resolved for imports, so the
    collected tests that import it run. ``imports`` stands for `import_map` when given: the
    imports of a tree as it was, for a selection replayed on it. ``first`` are the ids the
    last self-check found failing, on a fix-up; ``costs`` the seconds earlier self-checks
    timed test files at (`costs_from_record`), ``bounds`` those of them that are only lower
    bounds (`lower_bounds`); ``alone`` the files to run one per chunk (`run_alone`), kept
    in ``Selection.alone`` when the selection holds them.
    """
    from refactory_dsh.changed_functions import is_generated

    root = Path(root)
    files = ([] if imports is not None else repository_files(root)) if files is None else files
    scope = pytest_scope(check_command, root) if check_command else None
    last_failures = _failures(prev_record or {}, root)
    touched_tests: list[str] = []
    sources: list[str] = []
    gone: list[str] = []
    for name in dict.fromkeys(path.strip() for path in touched if path.strip()):
        if name.endswith(DOC_SUFFIXES) or is_generated(root, name):
            continue
        if is_test_file(name):
            if (root / name).is_file() and (not name.endswith(".py") or _in_scope(name, scope)):
                touched_tests.append(name)
            elif name.endswith(".py"):
                # Not a test this check runs, but a file a collected test may import: a
                # deleted test (it imports it and now fails at import), a test another
                # suite collects, or a product module named ``test_*.py`` (as
                # ``refactory_dsh/test_selection.py`` is). Its importers run; it does not.
                gone.append(name)
        else:
            sources.append(name)
    selection = Selection(touched_tests=sorted(touched_tests), last_failures=last_failures,
                          first=list(dict.fromkeys(runnable(test_id) for test_id in first
                                                   if (root / _file_of(test_id)).is_file())))
    if not sources and not touched_tests and not gone and not last_failures:
        selection.tests = list(dict.fromkeys(selection.first))
        selection.alone = _kept(alone, selection.tests)
        return selection
    if imports is None:
        imports = import_map(root, files, tuple(sources) + tuple(gone), scope)
    hub_set = hubs(imports)
    structural = sorted(test for test, modules in imports.items() if not modules)
    changed = set(sources) | set(gone)
    selection.hubs_changed = sorted(set(sources) & hub_set)
    if selection.hubs_changed:
        whole_engine = sorted((test for test, modules in imports.items()
                               if modules and hub_set <= modules),
                              key=lambda test: (len(imports[test]), test))
        selection.engine = [*structural, *whole_engine]
    importers = [test for test, modules in imports.items() if modules & changed]
    if sources and not selection.hubs_changed:
        selection.structural = structural
    rest = list(dict.fromkeys([*importers, *selection.structural]))
    cost = estimated_costs(rest, root, costs or {}, bounds)
    rest.sort(key=lambda test: (cost[test], test))
    selection.importers = [test for test in rest if test in importers]
    selection.structural = [test for test in rest if test not in importers]
    ordered: list[str] = list(dict.fromkeys(selection.first))
    whole: set[str] = {name for name in ordered if "::" not in name}
    for name in [*selection.engine, *selection.touched_tests]:
        if name not in whole:
            whole.add(name)
            ordered.append(name)
    for test_id in last_failures:
        if _file_of(test_id) not in whole and test_id not in ordered:
            ordered.append(test_id)
            if test_id == _file_of(test_id):
                whole.add(test_id)
    for name in rest:
        if name not in whole:
            whole.add(name)
            ordered.append(name)
    selection.tests = ordered
    selection.alone = _kept(alone, ordered)
    return selection


def _kept(alone: list[str], tests: list[str]) -> list[str]:
    """The files of ``alone`` that ``tests`` run, whole or by id."""
    files = {_file_of(name) for name in tests}
    return [name for name in dict.fromkeys(alone) if name in files]
