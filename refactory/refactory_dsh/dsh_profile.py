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
from pathlib import Path
from typing import NamedTuple

from refactory_dsh.paths import REPO_ROOT

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
GUI_PROFILE = "refactory-web"

#: The surface-independent layer: the `ceo` preset and the roster composition.
#: It is wanted under every surface, so both the GUI and the attached Martty
#: target carry it.
PRESETS_BUNDLE = "@deepseek-ai/dsh-refactory-presets"

#: The browser layer: the console, the right-sidebar picker and the brand rows.
#: It has no meaning under a TUI surface, so only the GUI target carries it.
WEB_SURFACE_BUNDLE = "@deepseek-ai/dsh-refactory-web"

#: The operator-owned TUI profile re:factory attaches the presets layer to.
MARTTY_PROFILE = "martty"

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
    """One ordered step that makes an owned target ready."""

    #: ``init`` initializes from a shipped template; ``add`` installs one layer.
    kind: str
    #: The target profile.
    profile: str
    #: The template name for ``init``; the checkout bundle directory for ``add``.
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


def dsh_home() -> Path:
    """The harness home that holds ``profiles/``; ``$DSH_HOME``, else ``~/.dsh``."""
    override = (os.environ.get("DSH_HOME") or "").strip()
    return Path(override) if override else Path.home() / ".dsh"


def profile_dir(name: str = GRAPH_PROFILE) -> Path:
    """Where the profile lives, whether or not it exists."""
    return dsh_home() / "profiles" / name


def target(name: str = GRAPH_PROFILE) -> Target:
    """The target a profile name names. Raises :class:`KeyError` when unowned."""
    for entry in TARGETS:
        if entry.profile == name:
            return entry
    raise KeyError(name)


def layer_bundle(bundle: str) -> str:
    """The bundle for a layer key, or the key itself when it is already a bundle."""
    entry = LAYERS.get(bundle)
    return entry.bundle if entry is not None else bundle


def declared_bundles(name: str = GRAPH_PROFILE) -> list[str]:
    """The bundles the profile's manifest declares, in order.

    Read from ``package.json`` rather than by importing anything: a profile that
    is absent and one that is present but does not carry the bundle are different
    failures, and telling them apart is what lets the error say which happened.
    """
    manifest = profile_dir(name) / "package.json"
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


def is_ready(name: str = GRAPH_PROFILE, bundle: str = GRAPH_BUNDLE) -> bool:
    """Whether a mode can be run on DSH through this profile right now."""
    return bundle in declared_bundles(name)


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


def provision_plan() -> list[ProvisionStep]:
    """The ordered steps that make every owned target ready.

    ``install.sh`` and ``./refactory-dsh`` both execute this rather than keeping
    their own copy of the profile list, so the composition has one owner.
    Attached targets are never part of the plan: re:factory does not create or
    mutate a profile it does not own.
    """
    steps: list[ProvisionStep] = []
    for entry in TARGETS:
        if not entry.owned:
            continue
        present = (profile_dir(entry.profile) / "package.json").exists()
        if not present and entry.base_template is not None:
            steps.append(ProvisionStep("init", entry.profile, entry.base_template))
        declared = declared_bundles(entry.profile)
        for key in entry.layers:
            bundle = LAYERS[key].bundle
            if bundle not in declared:
                steps.append(ProvisionStep("add", entry.profile, bundle_directory(bundle)))
    return steps


def render_plan(steps: list[ProvisionStep] | None = None) -> str:
    """The provisioning plan as tab-separated rows for the shell scripts."""
    chosen = provision_plan() if steps is None else steps
    return "".join(f"{step.kind}\t{step.profile}\t{step.value}\n" for step in chosen)


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


def readiness_gaps() -> list[tuple[str, str, str]]:
    """Every gap ``doctor`` should show.

    Owned targets always; an attached target only when the operator already has
    that profile, because a profile that does not exist is not a chosen
    re:factory path and must not read as a broken install.
    """
    gaps = missing_bundles()
    for entry in TARGETS:
        if entry.owned or not (profile_dir(entry.profile) / "package.json").exists():
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
    ours = [entry.bundle for entry in LAYERS.values()]
    found: list[tuple[str, list[str]]] = []
    for name in SHIPPED_TEMPLATES:
        carried = [bundle for bundle in declared_bundles(name) if bundle in ours]
        if carried:
            found.append((name, carried))
    return found


def migration_command(name: str, bundles: list[str]) -> str:
    """The command that removes stale re:factory layers from a harness profile."""
    return f"dsh plugin --profile {name} remove " + " ".join(bundles)


def requirement(name: str = GRAPH_PROFILE, bundle: str = GRAPH_BUNDLE) -> str | None:
    """Why DSH cannot start this target's work, and the commands that fix it.

    ``None`` when the profile carries the bundle. The reason is stated because the
    bare ``profile "graph" does not exist`` is indistinguishable from a packaging
    bug: the profile is not derived from the bundle, so a checkout that built and
    imported its package perfectly still has no profile until somebody runs the
    command.
    """
    entry = target(name)
    directory = profile_dir(name)
    if not (directory / "package.json").exists():
        if entry.owned and entry.base_template is not None:
            return (
                f"the DSH profile {name!r} does not exist at {directory}. It is a "
                f"re:factory-owned profile over the harness's {entry.base_template!r} "
                "template: create it once per machine with "
                f"`{init_command(name)}`, then add this layer with "
                f"`{add_command(name, bundle)}`"
            )
        if entry.owned:
            return (
                f"the DSH profile {name!r} does not exist at {directory}. A DSH "
                "profile is machine state, not something the bundle creates: create "
                f"it once per machine with `{add_command(name, bundle)}`"
            )
        return (
            f"the DSH profile {name!r} does not exist at {directory}. re:factory does "
            f"not create a profile it does not own: install the {name!r} stack first, "
            f"then add the layer with `{add_command(name, bundle)}`"
        )
    if bundle not in declared_bundles(name):
        return (
            f"the DSH profile {name!r} exists but does not carry {bundle}, so it is "
            f"not this checkout's {name!r} profile. Add it with `{add_command(name, bundle)}`"
        )
    return None


def require(name: str = GRAPH_PROFILE, bundle: str = GRAPH_BUNDLE) -> None:
    """Raise :class:`ProfileError` naming the fix and the reason, or return."""
    problem = requirement(name, bundle)
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
    "SHIPPED_TEMPLATES",
    "TARGETS",
    "WEB_SURFACE_BUNDLE",
    "WEB_TEMPLATE",
    "Layer",
    "Target",
    "add_command",
    "bundle_directory",
    "declared_bundles",
    "dsh_home",
    "init_command",
    "is_ready",
    "layer_bundle",
    "migration_command",
    "missing_bundles",
    "profile_dir",
    "provision_plan",
    "readiness_gaps",
    "render_plan",
    "require",
    "requirement",
    "stale_mutations",
    "target",
]
