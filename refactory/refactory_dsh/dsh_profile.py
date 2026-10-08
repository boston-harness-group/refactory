"""Which DSH profiles this checkout composes, whether they are ready, and how to fix them.

A DSH profile is **machine state**: ``dsh plugin --profile <name> add <package>``
writes ``$DSH_HOME/profiles/<name>`` with the bundle named in its
``dsh.profile.bundles``. The install path used to append re:factory's bundles to
the harness's shipped ``web`` template instead of composing a profile of its own.
That leaked re:factory's preset, console and brand rows into every
``dsh --profile web`` session and left no way to put the re:factory layer on a
different surface. This module owns the composition instead:

* a :class:`Layer` is one re:factory bundle, the surface it belongs to, and why;
* a :class:`Target` is one DSH profile, the shipped template it is initialized
  from, and the layers it carries. An **owned** target is created and healed by
  ``install.sh`` and ``./refactory-dsh``; an **attached** target belongs to the
  operator and is only ever reported with the command that adds a layer.

No harness-shipped profile template is written to. ``install.sh`` provisions the
owned targets only; ``refactory doctor`` reports a gap for every target (owned
always, attached only once the operator has that profile) with the exact command
that closes it, and :func:`stale_mutations` finds a shipped profile an older
install already mutated so the migration is stated rather than silent.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from refactory_dsh.paths import REPO_ROOT, dsh_root

#: The profile the DSH evaluator, the equivalence tests and a manual mode run use.
GRAPH_PROFILE = "graph"

#: The bundle that profile must carry. `@deepseek-ai/dsh-graph` is not published,
#: so it is installed from a checkout by path rather than by name.
GRAPH_BUNDLE = "@deepseek-ai/dsh-graph"

#: The shipped harness template the GUI profile is initialized from. Naming the
#: template rather than copying its bundle list keeps re:factory from drifting
#: when the harness adds a bundle to `web`.
WEB_TEMPLATE = "web"

#: The re:factory-owned profile the browser GUI boots. It is deliberately not the
#: harness's `web`: that profile belongs to the harness, and appending to it made
#: re:factory's rows appear in every unrelated `dsh --profile web` session.
GUI_PROFILE = "refactory"

#: The opt-in composition: the ordinary web bundles plus Refactory.
ADDITIVE_PROFILE = "refactory-web"

#: The surface-independent layer: the `ceo` preset and the roster composition.
#: It is wanted under every surface, so both the GUI and the attached Martty
#: target carry it.
PRESETS_BUNDLE = "@deepseek-ai/dsh-refactory"

#: The browser layer: the console, the right-sidebar picker and the brand rows.
#: It has no meaning under a TUI surface, so only the GUI target carries it.
WEB_SURFACE_BUNDLE = "@deepseek-ai/dsh-refactory-web"

#: The operator-owned TUI profile re:factory attaches the presets layer to.
MARTTY_PROFILE = "martty"

#: Bundles this checkout used to declare, mapped to the bundle that replaced
#: each. A profile manifest is machine state written by `dsh plugin add`, so an
#: installation that predates a rename keeps the retired name in its
#: `dependencies` and in its `dsh.profile.bundles`, and the bundle directory that
#: entry links to is gone. The plan removes it rather than leaving it dangling.
RETIRED_BUNDLES: dict[str, str] = {
    "@deepseek-ai/dsh-refactory-presets": PRESETS_BUNDLE,
}

#: Profile names this checkout used to compose, mapped to the profile that
#: replaced each. A DSH profile cannot be moved: its `node_modules` holds
#: absolute symlinks under `.dsh-module-fallback`, so renaming the directory
#: would leave every one of them pointing at the old path. The plan instead
#: retires re:factory's layers from the old directory, which leaves it an inert
#: copy of the shipped template it was initialized from rather than a profile
#: whose bundle entries dangle.
#: An explicitly configured additive profile with this name is active, not
#: retired; its refactoryAdditive flag keeps the legacy migration distinct.
RETIRED_PROFILES: dict[str, str] = {"refactory-web": GUI_PROFILE}

#: The harness's shipped profile templates, read for the stale-mutation check.
#: Kept local rather than imported from the harness so `doctor` works from a
#: plain clone; a template this list does not know about simply is not checked.
SHIPPED_TEMPLATES = ("acp", "web", "headless", "sdk", "sdk-minimal")


class Layer(NamedTuple):
    """One re:factory bundle, the surface it belongs to, and why it exists."""

    #: Short key the target tables use.
    key: str
    #: The bundle package name written into ``dsh.profile.bundles``.
    bundle: str
    #: ``any`` (every surface), ``web`` (a browser surface) or ``runtime``.
    surface: str
    #: Why the layer exists, in the words `doctor` and the docs use.
    why: str


class Target(NamedTuple):
    """One DSH profile, the template it extends, and the layers it carries."""

    #: The profile directory name under ``$DSH_HOME/profiles``.
    profile: str
    #: The shipped template an owned profile is initialized from, or None when
    #: the profile is a bare custom profile (`dsh plugin add` initializes it).
    base_template: str | None
    #: Layer keys to apply, in bundle order.
    layers: tuple[str, ...]
    #: True when re:factory creates and heals the profile; False when the
    #: operator owns it and re:factory only reports the command that adds a layer.
    owned: bool
    #: Why the target exists, in the words `doctor` and the docs use.
    why: str


class ProvisionStep(NamedTuple):
    """One ordered step that makes an owned target ready.

    ``value`` means one thing per :attr:`kind`, and the shells that execute
    :func:`render_plan` depend on that: ``init`` names a shipped template, ``add``
    names a directory under ``packages/bundle/`` that the caller prefixes with that
    root, ``remove`` names one or more packages to remove, space-separated, exactly
    as the manifest declares them, and ``clean`` names the obsolete selections a
    manifest rewrite has to drop. A ``remove`` is only ever emitted for a bundle
    the manifest also declares as a dependency: `dsh plugin remove` forwards to
    `pnpm remove`, which removes nothing at all when one of its names is not in
    that list.
    """

    #: ``init`` initializes from a shipped template; ``add`` installs one layer;
    #: ``remove`` retires declared dependencies; ``clean`` rewrites the selection
    #: list to drop what `pnpm remove` cannot reach.
    kind: str
    #: The target profile.
    profile: str
    #: The template name for ``init``; the checkout bundle directory for ``add``;
    #: the space-separated package names for ``remove`` and ``clean``.
    value: str


#: Every layer this checkout contributes, by key.
LAYERS: dict[str, Layer] = {
    "graph": Layer(
        "graph", GRAPH_BUNDLE, "runtime",
        "the workflow spine the Python glue runs modes on DSH",
    ),
    "presets": Layer(
        "presets", PRESETS_BUNDLE, "any",
        "the ceo preset and the roster default, wanted under every surface",
    ),
    "web": Layer(
        "web", WEB_SURFACE_BUNDLE, "web",
        "the browser console, right-sidebar picker and brand rows",
    ),
}

#: Every profile this checkout composes, in provisioning order.
TARGETS: tuple[Target, ...] = (
    Target(
        GRAPH_PROFILE, None, ("graph",), True,
        "the profile the Python glue runs modes on",
    ),
    Target(
        GUI_PROFILE, WEB_TEMPLATE, ("presets", "web"), True,
        "the browser GUI, a re:factory-owned profile over the harness's web template",
    ),
    Target(
        MARTTY_PROFILE, None, ("presets",), False,
        "the surface-independent layers on an operator-owned Martty profile",
    ),
)


class ProfileError(RuntimeError):
    """A DSH profile a run needs is absent. Names the fix and the reason."""


def dsh_home(home: str | Path | None = None) -> Path:
    """Installation/session home: explicit path, nonblank DSH_HOME, then ~/.dsh."""
    override = str(home).strip() if home is not None else ""
    override = override or (os.environ.get("DSH_HOME") or "").strip()
    return Path(override or Path.home() / ".dsh").expanduser().resolve()


def model_home(home: str | Path | None = None) -> Path:
    """Model home, resolved before containment changes HOME.

    An explicit path wins over nonblank REFACTORY_MODEL_DSH_HOME, then the
    installation/session home. Provisioning continues to use dsh_home().
    """
    override = str(home).strip() if home is not None else ""
    return dsh_home(override or (os.environ.get("REFACTORY_MODEL_DSH_HOME") or "").strip())


def profile_dir(name: str = GRAPH_PROFILE, *, home: str | Path | None = None) -> Path:
    """Where the profile lives, whether or not it exists."""
    return dsh_home(home) / "profiles" / name


def is_additive_profile(name: str) -> bool:
    """Distinguish the new opt-in profile from the old renamed standalone GUI."""
    try:
        manifest = json.loads((profile_dir(name) / "package.json").read_text())
        return manifest.get("dsh", {}).get("profile", {}).get("refactoryAdditive") is True
    except (OSError, ValueError, AttributeError):
        return False


def additive_target() -> Target:
    return Target(ADDITIVE_PROFILE, WEB_TEMPLATE, ("presets", "web"), True,
                  "the complete native web interface with Refactory added")


def target(name: str = GRAPH_PROFILE) -> Target:
    """The target a profile name names. Raises :class:`KeyError` when unowned."""
    if name == ADDITIVE_PROFILE and is_additive_profile(name):
        return additive_target()
    for entry in TARGETS:
        if entry.profile == name:
            return entry
    raise KeyError(name)


def layer_bundle(bundle: str) -> str:
    """The bundle for a layer key, or the key itself when it is already a bundle."""
    entry = LAYERS.get(bundle)
    return entry.bundle if entry is not None else bundle


def declared_bundles(name: str = GRAPH_PROFILE, *, home: str | Path | None = None) -> list[str]:
    """The bundles the profile's manifest declares, in order.

    Read from ``package.json`` rather than by importing anything: a profile that
    is absent and one that is present but does not carry the bundle are different
    failures, and telling them apart is what lets the error say which happened.
    """
    return declared_bundles_in(profile_dir(name, home=home))


def declared_bundles_in(directory: Path) -> list[str]:
    """The bundles a profile directory's manifest declares, or ``[]``."""
    manifest = directory / "package.json"
    if not manifest.exists():
        return []
    try:
        loaded = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, dict):
        return []
    dsh = loaded.get("dsh")
    profile = dsh.get("profile") if isinstance(dsh, dict) else None
    bundles = profile.get("bundles") if isinstance(profile, dict) else None
    if not isinstance(bundles, list):
        return []
    return [str(entry) for entry in bundles]


def declared_dependencies(name: str = GRAPH_PROFILE) -> list[str]:
    """The package names the profile manifest declares as dependencies.

    The profile manifest carries two lists with different meanings: ``dependencies``
    is the specifier list `pnpm` reads and writes, and ``dsh.profile.bundles`` is
    the selection list the harness composes. `pnpm remove` speaks the first, so a
    name the caller passes to it has to come from here, not from the selection
    list.
    """
    return declared_dependencies_in(profile_dir(name))


def declared_dependencies_in(directory: Path) -> list[str]:
    """The dependency names a profile directory's manifest declares, or ``[]``."""
    manifest = directory / "package.json"
    if not manifest.exists():
        return []
    try:
        loaded = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(loaded, dict):
        return []
    dependencies = loaded.get("dependencies")
    if not isinstance(dependencies, dict):
        return []
    return [str(entry) for entry in dependencies]


def retired_bundles(name: str) -> list[str]:
    """The retired bundle names this profile still declares, in manifest order.

    Empty once the profile is clean, which is what keeps the plan idempotent:
    ``dsh plugin remove`` fails on a package the profile does not declare, so a
    second removal must never be emitted.
    """
    return [bundle for bundle in declared_bundles(name) if bundle in RETIRED_BUNDLES]


def refactory_layers(name: str) -> list[str]:
    """Every re:factory bundle this profile declares, current or retired."""
    ours = {entry.bundle for entry in LAYERS.values()} | set(RETIRED_BUNDLES)
    return [bundle for bundle in declared_bundles(name) if bundle in ours]


def wanted_layers(name: str) -> set[str]:
    """The re:factory bundles this profile is supposed to carry.

    A retired profile is supposed to carry none: it is a directory the profile
    rename left behind, and the plan retires re:factory's layers out of it rather
    than leaving it a profile whose bundle entries dangle.
    """
    if name == ADDITIVE_PROFILE and is_additive_profile(name):
        return {PRESETS_BUNDLE, WEB_SURFACE_BUNDLE}
    if name in RETIRED_PROFILES:
        return set()
    entry = target(name)
    return {LAYERS[key].bundle for key in entry.layers}


def obsolete_selections(name: str) -> list[str]:
    """re:factory names this profile declares but is not supposed to carry.

    The one rule the plan's ``remove`` rows and the manifest cleanup both apply,
    so the two cannot disagree about what "obsolete" means. A name qualifies only
    when it is re:factory's own *and* is not a layer this profile wants: a current
    layer that is merely absent from ``dependencies`` is a missing install, not an
    obsolete selection, and deleting it would take away a layer the profile needs.
    """
    return _obsolete_names(name, declared_bundles(name))


def _obsolete_names(name: str, names: list[str]) -> list[str]:
    """The same retirement rule for selections and installed dependencies."""
    wanted = wanted_layers(name)
    ours = {entry.bundle for entry in LAYERS.values()} | set(RETIRED_BUNDLES)
    return [
        bundle
        for bundle in names
        if bundle in ours and bundle not in wanted
    ]


def is_ready(name: str = GRAPH_PROFILE, bundle: str = GRAPH_BUNDLE, *, home: str | Path | None = None) -> bool:
    """Whether a mode can be run on DSH through this profile right now."""
    return bundle in declared_bundles(name, home=home)


def bundle_directory(bundle: str) -> str:
    """The checkout directory holding a bundle package, from the bundle's NAME.

    Derived rather than looked up per profile, so a layer is a one-line change to
    :data:`LAYERS`/:data:`TARGETS` and cannot name the wrong directory: the old
    code took the directory from the *profile* name, which only ever worked
    because `graph` and its bundle happened to agree.
    """
    return bundle.rsplit("/", 1)[-1].replace("dsh-", "", 1)


def add_command(
    name: str = GRAPH_PROFILE,
    bundle: str | None = None,
    *,
    checkout: Path | None = None,
) -> str:
    """The command that installs one target layer, copy-pasteable, with the real path.

    ``bundle`` selects which of the target's layers the command installs; when it
    is omitted the target's first layer is used. Raises :class:`KeyError` for a
    profile this checkout does not compose — or for a ``(profile, bundle)`` pair
    it does not compose — rather than falling back to one of them: a command
    naming the wrong bundle directory reads as a plausible fix and is worse than
    no command at all.
    """
    root = Path(checkout) if checkout is not None else REPO_ROOT
    entry = target(name)
    chosen = bundle if bundle is not None else LAYERS[entry.layers[0]].bundle
    if chosen not in {LAYERS[key].bundle for key in entry.layers}:
        raise KeyError(chosen)
    directory = bundle_directory(chosen)
    return f"dsh plugin --profile {name} add {root / 'packages' / 'bundle' / directory}"


def init_command(name: str, *, checkout: Path | None = None) -> str:
    """The command that creates an owned target, before its layers are added.

    A target initialized from a shipped template must copy that template's bundle
    list first; ``dsh plugin add`` alone would initialize a bare profile and the
    surface's own bundles would never be listed. A target with no template is
    initialized by its first ``dsh plugin add``, so that is what is returned.
    """
    entry = target(name)
    if not entry.owned:
        raise KeyError(name)
    if entry.base_template is None:
        return add_command(name, checkout=checkout)
    return (
        f"dsh --profile {name} --from-default-profile {entry.base_template} --dump-config"
    )


def provision_plan(*, additive: bool = False) -> list[ProvisionStep]:
    """The ordered steps that make every owned target ready.

    ``install.sh`` and ``./refactory-dsh`` both execute this rather than keeping
    their own copy of the profile list, so the composition has one owner.
    With ``additive=True``, provision graph and the opt-in additive web target
    instead of the standalone GUI. Attached targets are never part of the plan:
    re:factory does not create or mutate a profile it does not own.

    A target's ``remove`` rows precede its ``add`` rows, so a rename retires the
    old entry before the replacement is appended. ``RETIRED_PROFILES`` is the
    second source: those profiles are not targets any more, and the target loop
    above only creates the profile that replaced them, so nothing else would clear
    re:factory's layers out of the old directory.

    A ``remove`` row names only explicitly obsolete declared dependencies,
    because `pnpm remove` fails as a whole when one name is missing
    from that list and removes nothing -- which aborted every row after it under
    the shell scripts' ``set -e``. What that leaves behind is a selection that is
    no longer a dependency at all: no `pnpm remove` can name it, and the harness's
    ``reconcile()`` keeps rather than drops it. A ``clean`` row runs first under
    the profile's writer lock, before an unresolved selection can break a plugin
    operation's reconcile. Obsolete dependencies are removed even if a previous
    run already cleared their selections and stopped before removing them.
    """
    steps: list[ProvisionStep] = []
    targets = (target(GRAPH_PROFILE), additive_target()) if additive else TARGETS
    for entry in targets:
        if not entry.owned:
            continue
        present = (profile_dir(entry.profile) / "package.json").exists()
        if not present and entry.base_template is not None:
            steps.append(ProvisionStep("init", entry.profile, entry.base_template))
        steps.extend(_retirement_steps(entry.profile, entry.layers))
    for retired in RETIRED_PROFILES:
        if additive and retired == ADDITIVE_PROFILE:
            continue
        layers = additive_target().layers if is_additive_profile(retired) else ()
        steps.extend(_retirement_steps(retired, layers))
    return steps


def _retirement_steps(profile: str, layers: tuple[str, ...]) -> list[ProvisionStep]:
    """The rows that retire what a profile carries but should not.

    The ``clean`` row comes first, before every ``dsh plugin`` row for this
    profile. A stale selection that is not resolvable makes the harness's
    ``reconcile()`` throw, and `reconcile()` runs after any plugin operation that
    exits 0 -- so a plugin row that runs while the stale entry is still selected
    fails on it before the cleanup has had a chance to clear it. Clearing the
    selection list first removes the failure mode rather than racing it.

    ``clean`` therefore must NOT hold the profile's writer lock across the plugin
    rows: those acquire the same lock themselves, so holding it would deadlock.
    It takes the lock for its own rewrite and releases it immediately.
    """
    obsolete = obsolete_selections(profile)
    declared = declared_dependencies(profile)
    removable = [
        bundle for bundle in _obsolete_names(profile, list(dict.fromkeys(obsolete + declared)))
        if bundle in declared
    ]
    steps: list[ProvisionStep] = []
    if obsolete:
        steps.append(ProvisionStep("clean", profile, " ".join(obsolete)))
    if removable:
        steps.append(ProvisionStep("remove", profile, " ".join(removable)))
    current = declared_bundles(profile)
    for key in layers:
        bundle = LAYERS[key].bundle
        if bundle not in current or bundle not in declared:
            steps.append(ProvisionStep("add", profile, bundle_directory(bundle)))
    return steps


def render_plan(steps: list[ProvisionStep] | None = None) -> str:
    """The provisioning plan as tab-separated rows for the shell scripts."""
    chosen = provision_plan() if steps is None else steps
    return "".join(f"{step.kind}\t{step.profile}\t{step.value}\n" for step in chosen)


#: The Node helper that rewrites a profile manifest, beside this module.
_MANIFEST_HELPER = Path(__file__).resolve().parent / "profile_manifest.mjs"


def clean_manifest(profile: str) -> dict[str, object]:
    """Drop this profile's obsolete selections from its manifest, under its lock.

    A selection that is no longer a declared dependency is unreachable by every
    command the plan can emit: `pnpm remove` will not name it, and the harness's
    ``reconcile()`` keeps a name that is absent from ``dependencies`` rather than
    dropping it. Cleanup runs before plugin rows, for current and formerly owned
    profiles only. A persistent sibling backup retains the exact pre-image.

    The rewrite happens inside the harness's own writer lock -- the Node helper
    calls the pinned package's ``withFileLock``/``writeFileAtomic``, the same pair
    ``dsh plugin`` takes around every profile mutation -- so it cannot overwrite a
    plugin operation running at the same time. The pre-image is written before the
    replace and its retained path is reported on stderr. Returns the helper's
    report, or ``{}`` when there was nothing to do.
    """
    if profile in SHIPPED_TEMPLATES or any(
        entry.profile == profile and not entry.owned for entry in TARGETS
    ) or not (
        profile in RETIRED_PROFILES
        or any(entry.profile == profile and entry.owned for entry in TARGETS)
    ):
        raise ProfileError(f"refusing to rewrite profile {profile!r}: re:factory does not own it")
    progress = profile_dir(profile) / "package.json"
    if not progress.exists():
        return {}
    obsolete = obsolete_selections(profile)
    if not obsolete:
        return {}
    node = shutil.which("node")
    if node is None:
        raise ProfileError(
            f"node is not on the PATH, so the manifest of profile '{profile}' "
            "cannot be cleaned; install Node and re-run"
        )
    with tempfile.TemporaryDirectory(prefix="refactory-clean-") as scratch:
        request = Path(scratch) / "request.json"
        request.write_text(
            json.dumps(
                {
                    "profileDir": str(profile_dir(profile).resolve()),
                    "anchor": str(dsh_root().resolve()),
                    "obsolete": _obsolete_names(
                        profile, [entry.bundle for entry in LAYERS.values()] + list(RETIRED_BUNDLES)
                    ),
                }
            )
        )
        finished = subprocess.run(
            [node, str(_MANIFEST_HELPER), str(request)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if finished.returncode != 0:
            raise ProfileError(
                f"the manifest of profile '{profile}' could not be cleaned: "
                f"{(finished.stderr or finished.stdout).strip()}"
            )
        report = json.loads(finished.stdout or "{}")
        if report.get("status") == "rewritten":
            print(
                f"re:factory: profile {profile!r}: cleared {', '.join(report['removed'])}; "
                f"backup: {report['backupPath']}",
                file=sys.stderr,
            )
        return report


def missing_bundles() -> list[tuple[str, str, str]]:
    """Every owned layer its target does not carry yet.

    Returns one ``(profile, bundle, reason)`` row per gap, in target order, and an
    empty list when this machine is ready. Both caller shapes want the same
    answer: ``install.sh`` executes the plan to close the gaps, and a person sees
    *which* target is short rather than a bare boolean.
    """
    gaps: list[tuple[str, str, str]] = []
    for entry in TARGETS:
        if not entry.owned:
            continue
        for key in entry.layers:
            bundle = LAYERS[key].bundle
            problem = requirement(entry.profile, bundle)
            if problem is not None:
                gaps.append((entry.profile, bundle, problem))
    return gaps


def retired_reason(name: str, bundle: str) -> str:
    """Why a retired bundle entry has to go, and the command that removes it.

    The plan retires this from the profiles re:factory owns; for an attached
    profile the removal is the operator's, so the reason carries the command the
    same way :func:`requirement` does.
    """
    replacement = RETIRED_BUNDLES[bundle]
    return (
        f"the DSH profile {name!r} still carries {bundle}, which this checkout "
        f"renamed to {replacement}, and the directory it links to is gone. Retire "
        f"it with `{migration_command(name, [bundle])}`"
    )


def readiness_gaps() -> list[tuple[str, str, str]]:
    """Every gap ``doctor`` should show.

    Owned targets always; an attached target only when the operator already has
    that profile, because a profile that does not exist is not a chosen
    re:factory path and must not read as a broken install. A retired bundle the
    profile still declares is a gap as well: the plan retires it from an owned
    profile, and an attached profile only ever gets the command.
    """
    gaps = missing_bundles()
    for entry in TARGETS:
        if not entry.owned and not (profile_dir(entry.profile) / "package.json").exists():
            continue
        for bundle in retired_bundles(entry.profile):
            gaps.append((entry.profile, bundle, retired_reason(entry.profile, bundle)))
        if entry.owned:
            continue
        for key in entry.layers:
            bundle = LAYERS[key].bundle
            problem = requirement(entry.profile, bundle)
            if problem is not None:
                gaps.append((entry.profile, bundle, problem))
    return gaps


def stale_mutations() -> list[tuple[str, list[str]]]:
    """Shipped harness profiles that still carry re:factory layers.

    An install before this composition appended the layers to the harness's own
    ``web`` profile. That profile belongs to the harness, so the migration is to
    remove the layers from it; :func:`migration_command` names the fix. Returning
    the rows instead of clearing them is deliberate — the operator's machine state
    is not rewritten silently.
    """
    ours = {entry.bundle for entry in LAYERS.values()} | set(RETIRED_BUNDLES)
    found: list[tuple[str, list[str]]] = []
    for name in SHIPPED_TEMPLATES:
        carried = [bundle for bundle in declared_bundles(name) if bundle in ours]
        if carried:
            found.append((name, carried))
    return found


def migration_command(name: str, bundles: list[str]) -> str:
    """The command that removes re:factory layers an older install left behind.

    Serves both a harness-owned profile an old install polluted and a bundle a
    rename retired from a profile this checkout no longer composes.
    """
    return f"dsh plugin --profile {name} remove " + " ".join(bundles)


def requirement(name: str = GRAPH_PROFILE, bundle: str = GRAPH_BUNDLE, *, home: str | Path | None = None) -> str | None:
    """Why DSH cannot start this target's work, and the commands that fix it.

    ``None`` when the profile carries the bundle. The reason is stated because the
    bare ``profile "graph" does not exist`` is indistinguishable from a packaging
    bug: the profile is not derived from the bundle, so a checkout that built and
    imported its package perfectly still has no profile until somebody runs the
    command.
    """
    entry = target(name)
    directory = profile_dir(name, home=home)
    prefix = f"DSH_HOME={shlex.quote(str(dsh_home(home)))} " if home is not None else ""
    if not (directory / "package.json").exists():
        if entry.owned and entry.base_template is not None:
            return (
                f"the DSH profile {name!r} does not exist at {directory}. It is a "
                f"re:factory-owned profile over the harness's {entry.base_template!r} "
                "template: create it once per machine with "
                f"`{prefix}{init_command(name)}`, then add this layer with "
                f"`{prefix}{add_command(name, bundle)}`"
            )
        if entry.owned:
            return (
                f"the DSH profile {name!r} does not exist at {directory}. A DSH "
                "profile is machine state, not something the bundle creates: create "
                f"it once per machine with `{prefix}{add_command(name, bundle)}`"
            )
        return (
            f"the DSH profile {name!r} does not exist at {directory}. re:factory does "
            f"not create a profile it does not own: install the {name!r} stack first, "
            f"then add the layer with `{prefix}{add_command(name, bundle)}`"
        )
    if bundle not in declared_bundles(name, home=home):
        return (
            f"the DSH profile {name!r} exists but does not carry {bundle}, so it is "
            f"not this checkout's {name!r} profile. Add it with `{prefix}{add_command(name, bundle)}`"
        )
    return None


def require(name: str = GRAPH_PROFILE, bundle: str = GRAPH_BUNDLE, *, home: str | Path | None = None) -> None:
    """Raise :class:`ProfileError` naming the fix and the reason, or return."""
    problem = requirement(name, bundle, home=home)
    if problem is not None:
        raise ProfileError(problem)


__all__ = [
    "GRAPH_BUNDLE",
    "GRAPH_PROFILE",
    "GUI_PROFILE",
    "LAYERS",
    "MARTTY_PROFILE",
    "PRESETS_BUNDLE",
    "ProfileError",
    "ProvisionStep",
    "RETIRED_BUNDLES",
    "RETIRED_PROFILES",
    "SHIPPED_TEMPLATES",
    "TARGETS",
    "WEB_SURFACE_BUNDLE",
    "WEB_TEMPLATE",
    "Layer",
    "Target",
    "add_command",
    "bundle_directory",
    "declared_bundles",
    "declared_bundles_in",
    "dsh_home",
    "model_home",
    "init_command",
    "is_ready",
    "layer_bundle",
    "migration_command",
    "missing_bundles",
    "profile_dir",
    "provision_plan",
    "readiness_gaps",
    "refactory_layers",
    "render_plan",
    "require",
    "requirement",
    "retired_bundles",
    "retired_reason",
    "stale_mutations",
    "target",
]
