"""Packages own modes. Nothing else does.

A mode's identity is ``<package>/<name>``. The package is the unit a person
imports, lists, enables or removes, and it is the unit the generator emits, so a
created mode cannot end up unowned.

The library defines what a package *is* and never knows what any package
*contains*: ``refactory_dsh`` can list, import and validate packages without
naming one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle is typing-only
    from .mode import Mode

_BARE = re.compile(r"^[a-z][a-z0-9_]*$")


class PackageError(ValueError):
    """A package or mode identifier is not valid."""


def _check(part: str, label: str) -> None:
    if not _BARE.match(part or ""):
        raise PackageError(
            f"{label} {part!r} must be lower-case letters, digits and underscores, "
            "starting with a letter"
        )


def mode_id(package: str, name: str) -> str:
    """The canonical ``<package>/<name>`` identifier, validated."""
    _check(package, "package")
    _check(name, "mode name")
    return f"{package}/{name}"


def split(identifier: str) -> tuple[str, str]:
    """Split an identifier, refusing a bare name.

    A bare name is refused rather than defaulted into a package: defaulting is
    how a flat namespace grows back, one convenient call at a time.
    """
    if "/" not in identifier:
        raise PackageError(
            f"{identifier!r} is not package-qualified; a mode must live in a "
            "package, e.g. 'mypackage/mymode'"
        )
    package, name = identifier.split("/", 1)
    _check(package, "package")
    _check(name, "mode name")
    return package, name


#: Who vouches for a package, and therefore how it may be run.
TRUST_TIERS = ("trusted", "generated")


@dataclass(frozen=True)
class Package:
    """A named collection of modes, plus the operation modules they call."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    modes: tuple[type["Mode"], ...] = ()
    ops: tuple[str, ...] = ()
    extends: str = ""
    #: Other packages this one imports from. Declared so `remove` can refuse to
    #: break a dependent, instead of leaving a package that fails at import time.
    requires: tuple[str, ...] = ()
    #: `trusted` ships and was reviewed; `generated` was written by a model this
    #: run. The tier decides how it may be executed, and it is recorded rather than
    #: inferred -- a directory name is not a trust decision.
    trust: str = "trusted"
    #: Where it came from, for a package that was imported.
    origin: str = ""
    #: Outer-loop strategies (``OuterLoop`` subclasses) the package contributes,
    #: each selected by its class-level ``name`` (``strategies.resolve``).
    strategies: tuple[type, ...] = ()

    def __post_init__(self) -> None:
        _check(self.name, "package")
        seen: set[str] = set()
        for mode in self.modes:
            name = str(getattr(mode, "name", "") or "")
            if not name:
                raise PackageError(
                    f"{self.name}: mode class {mode.__name__} needs a class-level 'name'"
                )
            _check(name, "mode name")
            if name in seen:
                raise PackageError(f"{self.name}: two modes are both named {name!r}")
            seen.add(name)
        for strategy in self.strategies:
            if not str(getattr(strategy, "name", "") or ""):
                raise PackageError(
                    f"{self.name}: strategy class {strategy.__name__} needs a class-level 'name'"
                )
        if self.extends:
            _check(self.extends, "extended package")
        for requirement in self.requires:
            _check(requirement, "required package")
        if self.trust not in TRUST_TIERS:
            raise PackageError(
                f"{self.name}: unknown trust tier {self.trust!r}; known: {list(TRUST_TIERS)}"
            )

    def ids(self) -> list[str]:
        return sorted(mode_id(self.name, mode.name) for mode in self.modes)

    def mode(self, name: str) -> type["Mode"]:
        for mode in self.modes:
            if mode.name == name:
                return mode
        known = ", ".join(sorted(m.name for m in self.modes)) or "none"
        raise PackageError(f"package {self.name!r} has no mode {name!r}; has: {known}")

    def describe(self) -> dict[str, object]:
        return {
            "package": self.name,
            "description": self.description,
            "version": self.version,
            "extends": self.extends,
            "requires": list(self.requires),
            "trust": self.trust,
            "origin": self.origin,
            "modes": [
                {
                    "id": mode_id(self.name, mode.name),
                    "name": mode.name,
                    "description": getattr(mode, "description", ""),
                }
                for mode in sorted(self.modes, key=lambda m: m.name)
            ],
            "ops": list(self.ops),
        }
