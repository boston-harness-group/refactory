"""Refuse to repoint a DSH profile that another checkout's live run depends on.

``install.sh`` points this machine's DSH profiles at the checkout it runs in, and a
machine has one set of profiles per DSH home. Installing a second clone therefore
repoints them silently, and a live run on the first checkout has its spine replaced
underneath it. This guard runs before any profile is written: it names the other
checkout and the live run, and says how to keep both checkouts side by side.

The live-run registry is the one the driver already keeps (``pin.register_run_on_driver``
/ ``pin.driver_live_runs``): a registered run is live while its invocation lock is held.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from refactory_dsh import pin
from refactory_dsh.dsh_profile import GUI_PROFILE, GRAPH_PROFILE, dsh_home, profile_dir

#: The profiles ``install.sh`` links. Others belong to other installs.
PROFILES = (GRAPH_PROFILE, GUI_PROFILE)

#: Skips the live-run refusal. The repoint line is still printed.
FORCE_ENV = "REFACTORY_INSTALL_FORCE_RELINK"


def _checkout_of(resolved: Path) -> Path | None:
    """The checkout a bundle symlink resolves into, or ``None`` for somewhere else."""
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part == "packages" and parts[index + 1:index + 2] == ("bundle",) and index:
            return Path(*parts[:index])
    return None


def profile_checkout(profile: str, *, home: str | Path | None = None) -> Path | None:
    """The checkout this profile's bundle symlinks resolve to, or ``None``.

    ``None`` when the profile, or its links, cannot be read: there is then nothing this
    install could repoint. A profile whose layers resolve to two different checkouts
    reports the first, which is enough to refuse.
    """
    directory = profile_dir(profile, home=home) / "node_modules" / "@deepseek-ai"
    try:
        links = sorted(directory.iterdir())
    except OSError:
        return None
    for link in links:
        try:
            if not link.is_symlink():
                continue
            resolved = link.resolve(strict=True)
        except (OSError, ValueError):
            continue
        checkout = _checkout_of(resolved)
        if checkout is not None:
            return checkout
    return None


def refusal(profile: str, other: Path, this: Path, run: str) -> str:
    return (
        f"install.sh: the DSH {profile!r} profile points at {other}, which has a live run "
        f"({run}). Relinking would repoint that run. Finish or stop it first, or install "
        f"this checkout with its own DSH home: DSH_HOME=~/.dsh-<name> ./install.sh "
        f"(or set {FORCE_ENV}=1 to relink anyway)."
    )


def repoint(profile: str, other: Path, this: Path) -> str:
    return (
        f"install.sh: repointing the DSH {profile!r} profile from {other} to {this}. "
        f"The other checkout's model-backed commands will need ./install.sh re-run there."
    )


def check(checkout: Path, profiles: tuple[str, ...] = PROFILES, *,
          home: str | Path | None = None, force: bool | None = None) -> tuple[int, list[str]]:
    """Lines to say before installing, and 1 when a profile must not be repointed."""
    this = Path(checkout).expanduser().resolve()
    if force is None:
        value = str(os.environ.get(FORCE_ENV, "")).strip().lower()
        force = value not in ("", "0", "false", "no", "off")
    lines: list[str] = []
    for profile in profiles:
        other = profile_checkout(profile, home=home)
        if other is None or other == this:
            continue
        live = pin.driver_live_runs(other)
        if live and not force:
            return 1, [refusal(profile, other, this, live[0])]
        lines.append(repoint(profile, other, this))
    return 0, lines


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m refactory_dsh.install_guard <checkout> [profile ...]",
              file=sys.stderr)
        return 2
    # The installer passes exactly the profiles its plan would write, so no profiles
    # means it changes nothing and there is nothing to guard.
    code, lines = check(Path(args[0]), tuple(args[1:]))
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":  # pragma: no cover - the CLI entry
    raise SystemExit(main())
