"""Path resolution for refactory-dsh.

This file lives at ``<repo>/refactory/refactory_dsh/paths.py``. The glue lives at
``refactory/``; DSH itself runs from the pinned install the root manifest declares,
or from a source checkout at the repo root while one is there. Nothing else hardcodes
an absolute path; everything asks here, with environment overrides for non-default
layouts.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

# this package's directory: …/<repo>/refactory/refactory_dsh/paths.py
_PKG_DIR = Path(__file__).resolve().parent
# the glue directory: …/<repo>/refactory
_GLUE_DIR = _PKG_DIR.parent
# the product repo (the DSH fork): …/<repo>
REPO_ROOT = _GLUE_DIR.parent


def dsh_root() -> Path:
    """The DSH checkout that runs the headless model client."""
    return Path(os.environ.get("DSH_ROOT") or REPO_ROOT)


def project_root() -> Path:
    """The project a command resolves packages against.

    One answer for every entry point, because the alternative is what happened
    before there was one: the console defaulted to the repo root, the loop and
    registry CLIs defaulted to the working directory, and the authoring commands
    took ``Path.cwd()``. A package created through one of them was therefore
    invisible to the next, and which one you got depended on where you stood.

    ``REFACTORY_PROJECT`` overrides it for a checkout that keeps its packages
    somewhere else.
    """
    return Path(os.environ.get("REFACTORY_PROJECT") or REPO_ROOT)


def project_for_nodes() -> Path:
    """The project a **node command** resolves packages against.

    A node command's working directory is the run it belongs to, not the checkout, so
    the runtime states the project out loud (`REFACTORY_PROJECT`, set by the graph
    bundle). Where it is unset — the walker, a test, a person at a shell — the working
    directory is the project, which is what it has always been.
    """
    return Path(os.environ.get("REFACTORY_PROJECT") or Path.cwd())


class CheckoutProjectError(ValueError):
    """A package write resolved to refactory's own checkout instead of a project."""


def refactory_checkouts(workdir: Path | None = None) -> list[Path]:
    """Refactory's own checkouts, which are never a run's project: this one
    (:data:`REPO_ROOT`), and the driver and pin the run's record names (``workdir``'s
    ``.refactory/run-config.json``; ``REFACTORY_WORKDIR`` when no workdir is given).
    """
    import json

    found = [REPO_ROOT.resolve()]
    run = workdir if workdir is not None else os.environ.get("REFACTORY_WORKDIR")
    if run:
        try:
            driver = json.loads((Path(run) / ".refactory" / "run-config.json").read_text()).get("driver")
        except (OSError, ValueError, AttributeError):
            driver = None
        if isinstance(driver, dict):
            for key in ("checkout", "pin_checkout"):
                value = driver.get(key)
                if isinstance(value, str) and value:
                    path = Path(value).expanduser().resolve()
                    if path not in found:
                        found.append(path)
    return found


def project_for_writes(project: Path | None = None, *, workdir: Path | None = None) -> Path:
    """The project a node may write packages into: ``project``, else
    :func:`project_for_nodes`, resolved -- refused when it is refactory's own checkout.

    A run that never said which project it serves falls back to the working directory,
    and a run launched from the checkout then writes its packages into the checkout,
    where every later run on that checkout imports them. Saying the project
    (``REFACTORY_PROJECT``, or launching from the project's directory) is the fix, so
    the refusal says that. A project that is some *other* refactory clone is a project
    like any other.
    """
    resolved = Path(project if project is not None else project_for_nodes()).expanduser().resolve()
    for checkout in refactory_checkouts(workdir):
        if resolved == checkout:
            raise CheckoutProjectError(
                f"refusing to write packages into refactory's own checkout {checkout}: that is "
                "not a project. Set REFACTORY_PROJECT to the project the packages belong to, "
                "or start the run from that project's directory")
    return resolved


def refactory_home() -> Path:
    """Refactory's own per-user directory: global memory, the project registry, run worktrees.

    Outside every project, because what lives here belongs to no one checkout.
    ``REFACTORY_HOME`` overrides it, which is also how the tests keep it out of the real
    home directory.
    """
    return Path(os.environ.get("REFACTORY_HOME") or Path.home() / ".refactory").expanduser()


#: The oldest Node 22 DSH supports.
NODE_22_MINIMUM = (22, 19)

#: Node binaries a system or package manager installs outside a version manager.
SYSTEM_NODES = (
    Path("/usr/local/bin/node"),
    Path("/usr/bin/node"),
    Path("/opt/homebrew/opt/node@22/bin/node"),
    Path("/usr/local/opt/node@22/bin/node"),
)


def node_bin() -> str:
    """The Node binary that runs DSH, as an absolute path so it works under a
    restricted command PATH. Order: ``NODE_BIN``, the newest installed Node 22 (at
    least 22.19), the current ``PATH``, then a bare ``node``.

    Node 22 is chosen even when the default ``node`` is newer: under Node 24.20 DSH
    aborts intermittently with no output (workers fail ``exit -6``, about one call
    in four when three run at once), and under Node 22 it does not.
    """
    env = os.environ.get("NODE_BIN")
    if env:
        return env
    found = _installed_node_22()
    if found is not None:
        return found
    on_path = shutil.which("node")
    if on_path:
        return on_path
    return "node"


def _installed_node_22() -> str | None:
    """The newest supported Node 22 that nvm, fnm, the system or ``PATH`` provides."""
    home = Path.home()
    managed = [
        *Path(os.environ.get("NVM_DIR") or home / ".nvm").glob("versions/node/v22.*/bin/node"),
        *Path(os.environ.get("FNM_DIR") or home / ".local" / "share" / "fnm")
        .glob("node-versions/v22.*/installation/bin/node"),
    ]
    probed = [*SYSTEM_NODES, *([Path(on_path)] if (on_path := shutil.which("node")) else [])]
    candidates = [(_installed_version(path), path) for path in managed if path.is_file()]
    candidates += [(_installed_version(path.resolve()) or _binary_version(str(path)), path)
                   for path in probed if path.is_file()]
    supported = [(version, path) for version, path in candidates
                 if version is not None and version[0] == 22 and version[:2] >= NODE_22_MINIMUM]
    return str(max(supported)[1]) if supported else None


def _installed_version(path: Path) -> tuple[int, int, int] | None:
    """The version an install names its directory after (nvm, fnm, volta, Homebrew's
    Cellar, a CI tool cache), so most Nodes are identified without being run."""
    for part in reversed(path.parts):
        version = _version(part)
        if version is not None:
            return version
    return None


def _version(text: str) -> tuple[int, int, int] | None:
    """A ``MAJOR.MINOR.PATCH`` version, with or without a leading ``v``, or None."""
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", text.strip())
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


@functools.lru_cache(maxsize=None)
def _binary_version(binary: str) -> tuple[int, int, int] | None:
    """What a Node binary reports with ``--version``; None when it cannot say."""
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return _version(out.stdout)


#: The CLI entry inside a DSH source checkout, relative to that checkout's root.
SOURCE_CLI_ENTRY = Path("apps") / "cli" / "src" / "bin.ts"

#: The CLI entry an install of the pinned harness provides, relative to the checkout root.
INSTALLED_CLI_ENTRY = Path("node_modules") / "@deepseek-ai" / "dsh" / "lib" / "bin.js"


def installed_dsh() -> Path:
    """The `dsh` launcher an install of the pinned harness provides.

    The repository declares `@deepseek-ai/dsh` as a dependency, so pnpm writes this
    launcher into `node_modules/.bin`. Returned whether or not it exists, so a caller
    that selected this branch can name the missing path in its own error.
    """
    return dsh_root() / "node_modules" / ".bin" / "dsh"


def dsh_command(*, root: Path | None = None, cli: str | None = None) -> list[str]:
    """The argv prefix that launches DSH, before DSH's own arguments.

    Three answers, in order. ``DSH_CLI`` names an executable outright, which is how a
    caller drives a pinned install the checkout does not resolve on its own. A source
    checkout — one that still holds ``apps/cli/src/bin.ts`` — runs its own CLI by
    preloading ``source_bootstrap.mjs``, which installs tsx and the source
    configuration before the entry runs. Anything else runs the installed harness's
    entry, which is the only launcher left once the source copy is gone.

    Both Node-launched answers run on :func:`node_bin`. The installed entry is run
    directly rather than through `node_modules/.bin/dsh`: that launcher runs whichever
    ``node`` is first on ``PATH``, which is how DSH came to run on a Node it aborts
    under while ``node_bin`` named a supported one.

    The source preload resolves tsx and its source configuration from the selected
    checkout. The child's working directory remains the caller's, so an agent can
    read and write its own item artifacts without launching from the checkout.
    """
    override = (cli if cli is not None else os.environ.get("DSH_CLI", "")).strip()
    selected_root = Path(root) if root is not None else dsh_root()
    if override:
        return [override]
    source = selected_root / SOURCE_CLI_ENTRY
    if source.exists():
        preload = Path(__file__).with_name("source_bootstrap.mjs").as_uri()
        return [node_bin(), "--import", preload, str(source.resolve())]
    entry = selected_root / INSTALLED_CLI_ENTRY
    if entry.exists():
        return [node_bin(), str(entry)]
    return [str(selected_root / "node_modules" / ".bin" / "dsh")]


def venv_bin() -> Path:
    """The venv whose python has refactory_dsh and the packages installed."""
    return Path(os.environ.get("VENV_BIN") or REPO_ROOT / ".venv" / "bin")


#: The spine bundle a built checkout provides, relative to the DSH root.
SPINE_LIB = Path("packages") / "workflow" / "workflow-spine" / "lib" / "index.js"


def require_build(root: Path | None = None, *, check_only: bool = False, gui_mode: bool = False) -> None:
    """Refuse immediately when the spine build is missing, and rebuild stale builds.

    Running from source (SOURCE_CLI_ENTRY exists) skips the check: tsx
    compiles on the fly. Otherwise, the built bundle at SPINE_LIB must
    exist. After the existence check, the workspace stamp is compared.
    When stale and ``gui_mode`` is true, ``RuntimeError`` is raised asking
    the person to restart.  When stale otherwise, ``pnpm run build`` is run
    under a lock and only when no live run uses the checkout.

    ``gui_mode`` should be passed as ``True`` only by the process that IS the
    GUI (the console).  Owners, supervisors and other children inherit
    ``REFACTORY_GUI_MODE`` from the environment but should NOT be refused a
    rebuild -- they pass the default ``gui_mode=False``.
    """
    selected = Path(root) if root is not None else dsh_root()
    if (selected / SOURCE_CLI_ENTRY).exists():
        return  # running from source; tsx handles it
    if not (selected / SPINE_LIB).exists():
        raise RuntimeError(
            "the workflow-spine build is missing; run `pnpm run build` "
            "from the checkout root"
        )
    # ── workspace stamp check ──
    stamp_file = selected / "packages" / "workflow" / "workflow-spine" / "lib" / ".workspace-stamp"
    if stamp_file.exists():
        from refactory_dsh.pin import workspace_stamp as _ws
        saved_stamp = stamp_file.read_text().strip()
        current_stamp = _ws(selected)
        if current_stamp is not None and saved_stamp != current_stamp:
            if gui_mode:
                raise RuntimeError(
                    "the build is stale; restart the GUI to pick up changes — "
                    "run `pnpm run build` and relaunch"
                )
            if check_only:
                import logging
                logging.getLogger(__name__).warning(
                    "the build is stale; run `pnpm run build` to rebuild"
                )
                return
            # Acquire the build lock BEFORE the live-run check to prevent a
            # TOCTOU race where a run starts between the check and the build.
            import fcntl
            import logging
            build_lock = refactory_home() / "build.lock"
            build_lock.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(str(build_lock), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                # Re-check stamps after acquiring the lock.
                saved_stamp2 = stamp_file.read_text().strip()
                current_stamp2 = _ws(selected)
                if current_stamp2 is not None and saved_stamp2 != current_stamp2:
                    # When a live run uses this checkout, warn but do not rebuild.
                    try:
                        _refuse_if_live_runs(selected)
                    except RuntimeError as exc:
                        logging.getLogger(__name__).warning(str(exc))
                        return
                    logging.getLogger(__name__).info("workspace build is stale; rebuilding")
                    try:
                        subprocess.run(["pnpm", "run", "build"], cwd=str(selected), check=True)
                    except subprocess.CalledProcessError:
                        logging.getLogger(__name__).warning(
                            "rebuild failed; the build is stale — run `pnpm run build` manually"
                        )
            finally:
                os.close(lock_fd)


def _refuse_if_live_runs(checkout: Path) -> None:
    """Raise if any live run uses this checkout as its driver (pinned or not).

    A run is live when its invocation lock is currently held, checked via
    :func:`run_ownership.held`.
    """
    from refactory_dsh.run_ownership import held as _run_held
    try:
        from refactory_dsh.pin import list_pins
        for pin_data in list_pins():
            if pin_data.get("driver_checkout") == str(checkout):
                for run_path_str in pin_data.get("runs", []):
                    rp = Path(run_path_str)
                    if rp.exists():
                        try:
                            if _run_held(rp):
                                raise RuntimeError(
                                    f"the build is stale but a live run ({run_path_str}) uses "
                                    f"this checkout; run `pnpm run build` manually after the run finishes"
                                )
                        except RuntimeError:
                            raise
                        except Exception:
                            pass
    except (ImportError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise
    # Check unpinned runs registered directly on this driver checkout.
    try:
        from refactory_dsh.pin import driver_live_runs
        live = driver_live_runs(checkout)
        if live:
            raise RuntimeError(
                f"the build is stale but a live run ({live[0]}) uses "
                f"this checkout; run `pnpm run build` manually after the run finishes"
            )
    except (ImportError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError):
            raise


def _source_manifest(root: Path) -> dict[str, str]:
    """Hash every regular file under *root*, excluding common generated directories."""
    ignored = {".git", ".factory", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in ignored or part.startswith(".env") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"source project contains a symlink: {relative}; use an explicit plain-file source")
        if path.is_file():
            result[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def python() -> Path:
    return venv_bin() / "python"
