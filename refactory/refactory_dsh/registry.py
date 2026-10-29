"""Discover, list and resolve packages of modes.

Discovery is a scan of the ``refactory_packages`` distribution for subpackages
that declare ``PACKAGE``. Nothing here names a package: the library can enumerate
and import every package without knowing that any particular one exists.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import structlog

from refactory_dsh.nn.package import Package, PackageError, mode_id, split
from refactory_dsh.paths import project_root

log = structlog.get_logger()

#: The distribution that holds packages, kept separate from the library so the
#: library cannot reach one by accident.
ROOT_PACKAGE = "refactory_packages"

#: Which packages a run may use, and which are switched off.
STATE_FILE = Path(".factory") / "packages.json"

#: What the writer records about where a package came from.
ORIGIN = ".origin.json"

#: Where generated packages land. The installed tree is read-only, so a run never
#: writes into the checkout; this is the writable side, and it is also the boundary
#: a sandbox needs.
WORKSPACE = Path(".factory") / "packages"

#: The ``.origin.json`` key a create or edit run records its smoke in: `SMOKE_PENDING`
#: from registration until the smoke passes, then `SMOKE_PASSED`. A package without the
#: key (installed, added, adopted, or registered before the key existed) is not marked.
SMOKE = "smoke"
SMOKE_PENDING = "pending"
SMOKE_PASSED = "passed"

#: How the listing marks a package whose smoke has not passed (dm-264).
UNVERIFIED = "unverified (smoke not passed)"


class PackageClash(PackageError):
    """A package name exists in both the install and the workspace."""


def workspace_root(project: Path) -> Path:
    """The writable package root for a project."""
    return Path(project) / WORKSPACE


def install_workspace(project: Path, *, project_override: Path | None = None) -> Path | None:
    """Make workspace packages importable under `refactory_packages`.

    Extending the distribution's `__path__` is what lets a package in a writable
    directory be imported by the same ``refactory_packages.<name>`` path the
    installed ones use, so an op written in a generated package needs no special
    case and the op namespace does not depend on where a package happens to live.

    The requested workspace takes priority over previously installed workspaces.
    ``project_override``, when given, takes priority over that workspace too, so a
    worktree's ``.factory/packages/`` is importable alongside the shared one.
    """
    import refactory_packages

    root = workspace_root(project)
    if not root.is_dir():
        if project_override is None:
            return None
    else:
        resolved = str(root.resolve())
        if resolved in refactory_packages.__path__:
            refactory_packages.__path__.remove(resolved)
        refactory_packages.__path__.insert(0, resolved)
        importlib.invalidate_caches()
    if project_override is not None:
        override_root = workspace_root(project_override)
        if override_root.is_dir():
            override_resolved = str(override_root.resolve())
            if override_resolved in refactory_packages.__path__:
                refactory_packages.__path__.remove(override_resolved)
            refactory_packages.__path__.insert(0, override_resolved)
            importlib.invalidate_caches()
            return override_root
    return root


def packages_root() -> Path:
    """The directory holding every package."""
    module = importlib.import_module(ROOT_PACKAGE)
    if not module.__file__:  # pragma: no cover - a namespace package has no dir
        raise PackageError(f"{ROOT_PACKAGE} is a namespace package; it needs a directory")
    return Path(module.__file__).parent


def _scan(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists() and not entry.name.startswith("_")
    ]


def discover_names(project: Path | None = None) -> list[str]:
    """Every package directory, installed or in the workspace.

    Both roots are scanned, and a name in both is refused rather than resolved:
    silent precedence is how a contributor's package gets shadowed and nobody can
    find out.
    """
    installed = set(_scan(packages_root()))
    if project is None:
        return sorted(installed)
    here = set(_scan(workspace_root(Path(project))))
    clash = sorted(installed & here)
    if clash:
        raise PackageClash(
            f"package(s) {clash} exist in both the install and {workspace_root(Path(project))}; "
            "one must be renamed"
        )
    return sorted(installed | here)


def where(name: str, project: Path | None = None) -> Path | None:
    """Which root a package lives in, or ``None`` if it is not there."""
    for root in ([workspace_root(Path(project))] if project else []) + [packages_root()]:
        if (root / name / "__init__.py").exists():
            return root / name
    return None


def load(name: str, project: Path | None = None) -> Package:
    """Import one package and return its manifest, with its provenance applied.

    A workspace package is evicted from the import cache first, because the workspace
    is mutable: create-mode rewrites it, and a regenerated package must never be
    judged by the module object the previous attempt left behind.
    """
    if project is not None:
        # A workspace package is only importable once the workspace is on the
        # distribution's path, and this is the one entry point that must not assume
        # a caller did it: a subprocess calling `load` directly is a real caller.
        install_workspace(Path(project))
        _evict_workspace(name, Path(project))
    try:
        module = importlib.import_module(f"{ROOT_PACKAGE}.{name}")
    except ImportError as exc:
        raise PackageError(f"cannot import package {name!r}: {exc}") from exc
    package = getattr(module, "PACKAGE", None)
    if not isinstance(package, Package):
        raise PackageError(
            f"package {name!r} does not declare PACKAGE = Package(...) in its __init__.py"
        )
    if package.name != name:
        raise PackageError(
            f"package {name!r} declares the name {package.name!r}; they must agree"
        )
    return _with_provenance(package, project)


def _evict_workspace(name: str, project: Path) -> None:
    """Drop a workspace package and its submodules from the import cache."""
    import sys

    root = workspace_root(project)
    if not (root / name / "__init__.py").exists():
        return
    module_name = f"{ROOT_PACKAGE}.{name}"
    for cached in [
        cached for cached in sys.modules
        if cached == module_name or cached.startswith(module_name + ".")
    ]:
        del sys.modules[cached]
    importlib.invalidate_caches()


def _origin_file(name: str, project: Path | None = None) -> Path | None:
    """The writer's record for a package, following the project it belongs to."""
    roots = [workspace_root(Path(project)) if project is not None else Path.cwd() / WORKSPACE]
    roots.append(packages_root())
    for root in roots:
        path = root / name / ORIGIN
        if path.exists():
            return path
    return None


def _with_provenance(package: Package, project: Path | None = None) -> Package:
    """Apply what the writer recorded, which outranks what the manifest claims.

    A manifest is code, and code can say anything. The tier a package is *run* under
    is the system's record of where it came from, so the record wins.
    """
    path = _origin_file(package.name, project)
    if path is None:
        return package
    try:
        recorded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return package
    if not isinstance(recorded, dict):
        return package
    from dataclasses import replace as _replace

    changes = {}
    if recorded.get("trust") in ("trusted", "generated"):
        changes["trust"] = recorded["trust"]
    if recorded.get("origin"):
        changes["origin"] = str(recorded["origin"])
    return _replace(package, **changes) if changes else package


def provenance(name: str, project: Path | None = None) -> dict[str, Any]:
    """Where a package came from, as the writer recorded it."""
    path = _origin_file(name, project)
    if path is None:
        return {"package": name, "trust": "trusted", "origin": "installed", "at": ""}
    try:
        recorded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        recorded = {}
    return {"package": name, **recorded}


def smoke_state(name: str, project: Path | None = None) -> str:
    """What was recorded about a package's smoke (`SMOKE`): by the create or edit run that
    registered it, and by a later passing `package smoke`.

    `SMOKE_PASSED` when nothing was recorded: only a run that registered the package
    writes the key, so a package that never went through one has nothing to verify.
    """
    path = _origin_file(name, project)
    return _smoke_in(path) if path is not None else SMOKE_PASSED


def recorded_smoke(root: Path) -> str:
    """`smoke_state` for the package at ``root``."""
    return _smoke_in(Path(root) / ORIGIN)


def _smoke_in(path: Path) -> str:
    try:
        recorded = json.loads(path.read_text()) if path.is_file() else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        recorded = {}
    return str(recorded.get(SMOKE, SMOKE_PASSED)) if isinstance(recorded, dict) else SMOKE_PASSED


def record_smoke(root: Path, status: str) -> None:
    """Record a smoke of the package at ``root`` (`SMOKE`)."""
    path = Path(root) / ORIGIN
    try:
        recorded = json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        recorded = {}
    if not isinstance(recorded, dict):
        recorded = {}
    recorded[SMOKE] = status
    path.write_text(json.dumps(recorded, indent=2, default=str))


def discover(project: Path | None = None) -> dict[str, Package]:
    """Every package that loads. A package that fails to import is reported."""
    found: dict[str, Package] = {}
    if project is not None:
        install_workspace(Path(project))
    for name in discover_names(project):
        try:
            found[name] = load(name, project)
        except PackageError as exc:
            log.warning("package_unavailable", package=name, reason=str(exc))
    return found


def _state(project: Path) -> dict[str, Any]:
    path = Path(project) / STATE_FILE
    if not path.exists():
        return {"packages": {}}
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"packages": {}}
    return loaded if isinstance(loaded, dict) else {"packages": {}}


def _save_state(project: Path, data: dict[str, Any]) -> None:
    path = Path(project) / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def is_enabled(project: Path, name: str) -> bool:
    entry = _state(Path(project)).get("packages", {}).get(name)
    return True if entry is None else bool(entry.get("enabled", True))


def set_enabled(project: Path, name: str, enabled: bool) -> None:
    state = _state(Path(project))
    entry = state.setdefault("packages", {}).setdefault(name, {})
    entry["enabled"] = enabled
    _save_state(Path(project), state)


def installed_names(project: Path | None = None) -> list[str]:
    """Package directories on disk, enabled or not."""
    return discover_names(project)


def enabled_packages(project: Path) -> dict[str, Package]:
    return {
        name: package
        for name, package in discover(Path(project)).items()
        if is_enabled(Path(project), name)
    }


def resolve(project: Path, identifier: str) -> tuple[Package, type]:
    """The package and mode class for ``<package>/<name>``."""
    package_name, mode_name = split(identifier)
    install_workspace(Path(project))
    if not is_enabled(Path(project), package_name):
        raise PackageError(f"package {package_name!r} is disabled for this run")
    package = load(package_name, project)
    return package, package.mode(mode_name)


def catalogue(project: Path) -> dict[str, list[dict[str, Any]]]:
    """Modes grouped by package: the only way they are ever listed."""
    project = Path(project)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, package in discover(project).items():
        entry = package.describe()
        entry["enabled"] = is_enabled(project, name)
        root = where(name, project)
        entry["where"] = str(root) if root else ""
        grouped[name] = entry["modes"]  # type: ignore[assignment]
        for mode in grouped[name]:
            mode["package_enabled"] = entry["enabled"]
            mode["trust"] = entry["trust"]
            mode["where"] = entry["where"]
    return grouped


def render_catalogue(project: Path) -> str:
    """The grouped listing a user sees.

    For a person: what each mode is, and what stands in its way. How an agent runs a
    mode (the console verb, ``wf_open``, the gates that ask a person) is in the JSON
    catalogue (`refactory list --json`), not here.
    """
    lines: list[str] = []
    for name in installed_names(Path(project)):
        try:
            package = load(name, Path(project))
        except PackageError as exc:
            lines.append(f"{name}/  (unavailable: {exc})")
            continue
        state = "" if is_enabled(Path(project), name) else "  (disabled)"
        marks = [] if package.trust == "trusted" else [package.trust]
        if not is_enabled(Path(project), name):
            marks.append("disabled")
        # A package a create or edit run registered is listed from that moment, because
        # its smoke has to import it; until the smoke passes it is a candidate, not a
        # mode that is known to run (dm-264).
        if smoke_state(name, Path(project)) != SMOKE_PASSED:
            marks.append(UNVERIFIED)
        # A declared tool this machine lacks is the one thing worth saying in a
        # listing: it is why a mode that is otherwise fine will not run here.
        needed = missing_tools(name, Path(project))
        if needed:
            marks.append("needs " + ", ".join(needed))
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        lines.append(f"{name}/  {package.description}{suffix}")
        for mode in sorted(package.modes, key=lambda m: m.name):
            identifier = mode_id(name, mode.name)
            description = getattr(mode, "description", "")
            lines.append(f"  {identifier:<38} {description}")
    return "\n".join(lines) + "\n" if lines else "no packages installed\n"


def smoke(project: Path, identifier: str, *, model: bool = False, client: Any = None) -> dict[str, Any]:
    """Run one mode once, so a package can be checked rather than merely imported.

    ``model`` supplies a model client to the run: a mode with an `agent` node cannot be
    scored without one, and most modes have one. ``client`` is that client, when the
    caller already settled its route; otherwise one is built on the stored choice.
    """
    import tempfile

    package, mode_class = resolve(Path(project), identifier)
    mode = mode_class()
    if not model:
        client = None
    elif client is None:
        from refactory_dsh.model import DshModelClient

        client = DshModelClient()
    with tempfile.TemporaryDirectory(prefix="refactory-smoke-") as tmp:
        result = mode.evaluate(Path(tmp), model=client)
    # A person who fixed a package whose run's smoke failed clears the listing's mark by
    # smoking it here; a package with nothing recorded is left as it is (dm-264).
    if result.exit_ok and smoke_state(package.name, Path(project)) != SMOKE_PASSED:
        root = where(package.name, Path(project))
        if root is not None and (root / ORIGIN).is_file():
            record_smoke(root, SMOKE_PASSED)
    return {
        "mode": identifier,
        "package": package.name,
        "exit_ok": result.exit_ok,
        "error": result.error,
        "score": result.score,
        # A search, `best` and `adopt` report a score with the search's cost penalty taken
        # off; this one has none, so the same mode reads higher here.
        "score_note": (f"{format(round(float(result.score), 6), 'g')} raw: the mode's own score, "
                       "without a search's cost penalty"
                       if isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
                       else "not scored"),
        "iterations": result.iterations,
    }


# ── the lifecycle ──────────────────────────────────────────────────────────
def _write_origin(
    root: Path, *, origin: str, trust: str, ref: str = "", source_name: str = "",
) -> Path:
    import datetime

    path = root / ORIGIN
    try:
        previous = json.loads(path.read_text()) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        previous = None
    path.write_text(json.dumps({
        **({"source_provenance": previous} if isinstance(previous, dict) else {}),
        "origin": origin,
        "ref": ref,
        **({"source_package": source_name} if source_name else {}),
        "trust": trust,
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    return path


def add(project: Path, source: Path, *, as_name: str = "", trust: str = "generated") -> str:
    """Copy a package into the workspace, refusing a clash.

    A copy rather than a link, so the added package keeps working if the source is
    moved. The name is refused if it is already taken: the old registry resolved a
    clash by source precedence, silently, which is how a contributor's package gets
    shadowed and nobody can find out.

    A refusal is atomic: the copy is undone, so the workspace is exactly what it was
    before the call. A half-added package would turn the next attempt into a false
    name clash.
    """
    import shutil

    from refactory_dsh import package_rewrite

    project = Path(project)
    source = Path(source).resolve()
    if not (source / "__init__.py").exists():
        raise PackageError(f"{source} is not a package: it has no __init__.py")
    destination_root = workspace_root(project)
    if destination_root.is_dir() and source.is_relative_to(destination_root.resolve()):
        raise PackageError(f"{source} is already in the workspace")
    original = package_rewrite.declared_name(source)
    if as_name and not original:
        raise PackageError(
            "--as needs a direct literal PACKAGE name; dynamic or ambiguous identities "
            "must be prepared under the new name before importing"
        )
    name = as_name or original or source.name
    _check_name(name)
    existing = where(name, project)
    if existing is not None:
        raise PackageClash(
            f"package {name!r} already exists at {existing}; pass --as to rename it"
        )
    target = destination_root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise PackageClash(f"destination {target} already exists; no files were changed") from exc
    try:
        # The exclusive mkdir establishes ownership before cleanup can remove it.
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if original and name != original:
            package_rewrite.rename(target, original, name)
        _write_origin(target, origin=str(source), trust=trust, source_name=original)
        install_workspace(project)
        # `load` refuses a package whose declared name differs, so it both checks
        # the copy and names the mismatch.
        load(name, project)
    except Exception:
        # The copy is before the check, so a refusal has to undo it: the residue
        # would make the next attempt a false name clash, and the clash message
        # sends the caller to `--as`, which is what they already passed.
        _evict_workspace(name, project)
        shutil.rmtree(target, ignore_errors=True)
        raise
    importlib.invalidate_caches()
    log.info("package_added", package=name, source=str(source), trust=trust)
    return name


def import_package(
    project: Path, source: str, *, ref: str = "", as_name: str = "", trust: str = "generated"
) -> str:
    """Bring a package in from a path or a git URL."""
    candidate = Path(source).expanduser()
    if candidate.exists():
        if ref:
            raise PackageError("--ref only applies to a git URL")
        return add(project, candidate, as_name=as_name, trust=trust)
    if not source.startswith(("http://", "https://", "git@")):
        raise PackageError(
            f"{source!r} is neither a path that exists nor a git URL; a package is a "
            "directory, so an import needs somewhere to copy it from"
        )
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="refactory-import-") as tmp:
        command = ["git", "clone", "--depth", "1"]
        if ref:
            command += ["--branch", ref]
        command += [source, tmp]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise PackageError(f"git clone failed: {result.stderr.strip()[-400:]}")
        name = add(project, Path(tmp), as_name=as_name, trust=trust)
        path = workspace_root(project) / name / ORIGIN
        recorded = json.loads(path.read_text())
        origin = source
        if source.startswith(("http://", "https://")):
            from urllib.parse import urlsplit, urlunsplit

            url = urlsplit(source)
            # Repository identity is useful provenance; transport credentials are not.
            origin = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
        recorded.update(origin=origin, ref=ref)
        path.write_text(json.dumps(recorded, indent=2))
        return name


def remove(project: Path, name: str, *, force: bool = False) -> Path:
    """Delete a workspace package, refusing to break a dependent.

    Only the workspace is writable. The installed tree is read-only, so removing a
    shipped package is refused rather than attempted.
    """
    import shutil

    project = Path(project)
    target = where(name, project)
    if target is None:
        raise PackageError(f"no package {name!r}")
    if not target.is_relative_to(workspace_root(project)):
        raise PackageError(
            f"{target} is in the install tree, which is read-only; only workspace "
            "packages can be removed"
        )
    if not force:
        dependents = sorted(
            other
            for other in discover_names(project)
            if other != name and name in load(other, project).requires
        )
        if dependents:
            raise PackageError(
                f"{dependents} require {name!r}; remove them first, or pass --force"
            )
    shutil.rmtree(target)
    state = _state(project)
    state.get("packages", {}).pop(name, None)
    _save_state(project, state)
    importlib.invalidate_caches()
    log.info("package_removed", package=name)
    return target


# ── what a package needs that is not Python ────────────────────────────────
def tools(name: str, project: Path | None = None):
    """The tools a package declares, read without importing it.

    Delegated to :mod:`refactory_dsh.toolchain`, which owns the declaration. The
    registry is the reader here because it is the thing users already ask "what is
    this package?" of, and because the answer has to be available for a package
    that does not import -- the case a missing tool creates.
    """
    from refactory_dsh import toolchain

    return toolchain.read_tools(name, project)


def tool_report(
    project: Path | None = None, name: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Every declared tool of every package, with its status on this machine.

    Separate from :func:`verify` on purpose. A missing tool is a fact about the
    machine, not a defect of the package: the package is correct, listable and
    runnable-in-principle, and reporting it as a verification failure would make
    "this checkout has no Lean" indistinguishable from "this package is broken".
    A *malformed declaration* is a package defect, and that one does appear in
    ``verify``.
    """
    from refactory_dsh import toolchain

    project = Path(project) if project is not None else None
    report: dict[str, list[dict[str, Any]]] = {}
    for package_name in ([name] if name else discover_names(project)):
        try:
            report[package_name] = [
                status.to_dict() for status in toolchain.report(package_name, project)
            ]
        except toolchain.ToolDeclError as exc:
            report[package_name] = [{"name": "", "error": str(exc)}]
    return report


def missing_tools(name: str, project: Path | None = None) -> list[str]:
    """The names of required tools this machine does not have, for a listing."""
    from refactory_dsh import toolchain

    try:
        return [
            status.name
            for status in (status for status in toolchain.report(name, project))
            if not status.present and not status.tool.optional
        ]
    except toolchain.ToolchainError:
        return []


def verify(project: Path, name: str | None = None) -> dict[str, list[str]]:
    """Import every mode of a package and say what is wrong with it.

    A tool that is *absent* is not a finding here; see :func:`tool_report`. A
    tool *declaration* that cannot be read is, because that is a defect of the
    package rather than of the machine it landed on.
    """
    from refactory_dsh import toolchain
    from refactory_dsh.nn import validate_state

    project = Path(project)
    findings: dict[str, list[str]] = {}
    for package_name in ([name] if name else discover_names(project)):
        problems: list[str] = []
        try:
            toolchain.read_tools(package_name, project)
        except toolchain.ToolDeclError as exc:
            problems.append(str(exc))
        try:
            package = load(package_name, project)
        except PackageError as exc:
            findings[package_name] = problems + [str(exc)]
            continue
        for required in package.requires:
            if where(required, project) is None:
                problems.append(f"requires {required!r}, which is not installed")
        for mode_class in package.modes:
            identifier = f"{package_name}/{mode_class.name}"
            try:
                mode = mode_class()
            except Exception as exc:
                problems.append(f"{identifier}: cannot be constructed: {exc}")
                continue
            for issue in validate_state(
                mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
            ):
                problems.append(f"{identifier}: {issue}")
        findings[package_name] = problems
    return findings


def _check_name(name: str) -> None:
    from refactory_dsh.nn.package import _check as _validate

    _validate(name, "package")


def describe_mode(project: Path, identifier: str) -> dict[str, Any]:
    """One mode's surface, for a person about to optimize it or read a report."""
    package, mode_class = resolve(Path(project), identifier)
    mode = mode_class()
    described = mode.describe()
    described["package"] = package.name
    described["id"] = identifier
    described["readme"] = package.description
    described["tools"] = tool_report(Path(project), package.name)[package.name]
    return described


def main(argv: list[str] | None = None) -> int:
    """``python -m refactory_dsh.registry list | show <mode>``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m refactory_dsh.registry",
        description="List packages of modes, or describe one mode.",
    )
    parser.add_argument(
        "verb",
        choices=["list", "show", "verify", "smoke", "tools", "add", "import", "remove",
                 "enable", "disable"],
        nargs="?", default="list",
    )
    parser.add_argument("target", nargs="?", help="a <package>/<name> mode, or a package")
    parser.add_argument("--project", default=str(project_root()))
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--model", action="store_true",
        help="for 'smoke': score a mode that calls a model, by supplying one",
    )
    parser.add_argument("--as", dest="as_name", default="", help="rename on add or import")
    parser.add_argument("--ref", default="", help="a git ref, for import")
    parser.add_argument("--force", action="store_true", help="override a refusal")
    args = parser.parse_args(argv)
    project = Path(args.project)

    def fail(message: str) -> int:
        print(json.dumps({"error": message}), file=sys.stderr)
        return 2

    try:
        if args.verb == "list":
            print(
                json.dumps(catalogue(project), indent=2, default=str)
                if args.json
                else render_catalogue(project),
                end="\n" if args.json else "",
            )
            return 0
        if args.verb == "show":
            if not args.target:
                return fail("'show' needs a <package>/<name> mode")
            print(json.dumps(describe_mode(project, args.target), indent=2, default=str))
            return 0
        if args.verb == "verify":
            findings = verify(project, args.target or None)
            print(json.dumps(findings, indent=2, default=str))
            return 1 if any(findings.values()) else 0
        if args.verb == "tools":
            report = tool_report(project, args.target or None)
            print(json.dumps(report, indent=2, default=str))
            unusable = [
                entry
                for entries in report.values()
                for entry in entries
                if entry.get("error") or (entry.get("present") is False and not entry.get("optional"))
            ]
            return 1 if unusable else 0
        if args.verb == "smoke":
            if not args.target:
                return fail("'smoke' needs a <package>/<name> mode")
            client = None
            if args.model:
                from refactory_dsh.model import DshModelClient, ModelChoiceError, require_model_choice

                try:
                    # The smoke's agent nodes call a model: settle its route first (dm-258).
                    client = require_model_choice(DshModelClient(), interactive=sys.stdin.isatty())
                except ModelChoiceError as exc:
                    print(f"refactory: {exc}", file=sys.stderr)
                    return 1
            result = smoke(project, args.target, model=bool(args.model), client=client)
            print(json.dumps(result, indent=2, default=str))
            return 0 if result["exit_ok"] else 1
        if args.verb == "add":
            if not args.target:
                return fail("'add' needs a path to a package directory")
            print(json.dumps({"added": add(project, Path(args.target), as_name=args.as_name)}))
            return 0
        if args.verb == "import":
            if not args.target:
                return fail("'import' needs a path or a git URL")
            imported = import_package(
                project, args.target, ref=args.ref, as_name=args.as_name
            )
            print(json.dumps({"imported": imported}))
            return 0
        if args.verb == "remove":
            if not args.target:
                return fail("'remove' needs a package name")
            removed = remove(project, args.target, force=args.force)
            print(json.dumps({"removed": str(removed)}))
            return 0
        if args.verb in ("enable", "disable"):
            if not args.target:
                return fail(f"'{args.verb}' needs a package name")
            set_enabled(project, args.target, args.verb == "enable")
            print(json.dumps({args.verb + "d": args.target}))
            return 0
    except PackageError as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
