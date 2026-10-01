"""Compare the harness this repository pins with the harness that serves the profile.

Every other check in this repository exercises re:factory's bundles against the
**pinned** harness — the one ``node_modules/.bin/dsh`` resolves to, which
``install.sh`` and ``pnpm run test:acceptance`` use. The profile is booted by
whatever ``dsh`` is on ``PATH``, which on a development machine is often a
different tree. Nothing compared the two, so a contract that differs between them
shipped green: ``e459e32637`` changed the strict ``TypertCodec`` from
``schema`` to ``create()`` in a harness whose ``package.json`` still says
``0.1.6-alpha.1``, and the mismatch surfaced only as a boot-failure page.

The version string alone is not enough, because that is exactly the blind spot:
the fork's tree and the published ``0.1.6-alpha.1`` both report
``0.1.6-alpha.1``. So this module compares the versions **and**, when both
harnesses' ``@deepseek-ai/dsh-typert-protocol`` package can be located, the strict
codec face each one declares. A disagreement names both versions and both faces.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from refactory_dsh.paths import installed_dsh

#: The strict-codec declaration each harness generation pins, by the field name
#: it declares. `schema` is the released contract through `0.1.6-alpha.1`;
#: `create` is what `e459e32637` introduced, first released in `0.1.6-alpha.2`.
_FACE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("create", re.compile(r"readonly\s+create\s*:")),
    ("schema", re.compile(r"readonly\s+schema\s*:")),
)

#: Environment override for the harness that serves the profile. Without it the
#: `dsh` on `PATH` is used; a value that names no launcher is reported, not
#: skipped, because it was set deliberately.
SERVING_DSH_ENV = "REFACTORY_DSH"


@dataclass(frozen=True)
class Harness:
    """One harness installation: where it lives, its version, its codec face."""

    #: Human label used in the disagreement message ("pinned", "serving").
    label: str
    #: The launcher that was inspected.
    launcher: Path
    #: The version ``dsh --version`` reported, or None when it did not answer.
    version: str | None
    #: The strict TypertCodec face its protocol package declares, or None.
    codec_face: str | None

    def describe(self) -> str:
        """One line naming the launcher, version and codec face."""
        version = self.version or "unknown version"
        face = self.codec_face or "unknown strict codec"
        return f"{self.label} harness {version} at {self.launcher} (strict TypertCodec: {face})"


@dataclass(frozen=True)
class HarnessCheck:
    """The two harnesses and why they disagree, if they do."""

    #: The harness the repository's dependency resolves to.
    pinned: Harness
    #: The harness on ``PATH`` (or ``REFACTORY_DSH``), absent when none resolves.
    serving: Harness | None
    #: The reason the two disagree; None when they agree or no separate serving
    #: harness exists.
    reason: str | None

    @property
    def ok(self) -> bool:
        """Whether the two harnesses agree (or only the pinned one exists)."""
        return self.reason is None


def pinned_launcher() -> Path:
    """The launcher of the harness this repository pins."""
    return installed_dsh()


def serving_launcher() -> Path | None:
    """The `dsh` that boots the profile: ``REFACTORY_DSH``, else ``PATH``."""
    override = (os.environ.get(SERVING_DSH_ENV) or "").strip()
    if override:
        return Path(override)
    found = shutil.which("dsh")
    return Path(found) if found else None


def _version(launcher: Path, *, timeout: int = 30) -> str | None:
    """The version the launcher reports, or None when it cannot answer."""
    try:
        done = subprocess.run(
            [str(launcher), "--version"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = f"{done.stdout or ''}{done.stderr or ''}".strip()
    return text.splitlines()[0].strip() if text else None


def protocol_package(launcher: Path) -> Path | None:
    """The ``@deepseek-ai/dsh-typert-protocol`` package directory this launcher uses.

    A published install nests the package under ``node_modules`` (hoisted, or in
    pnpm's ``.pnpm`` store); a monorepo checkout keeps it in
    ``packages/typert/protocol``. Both are found by walking outward from the
    launcher's real location, so a harness that replaces the launcher with a
    symlink or a wrapper is still located. Returns None when neither layout
    exists, which the caller reports as "face unknown" rather than guessing.
    """
    real = launcher.resolve()
    seen: set[str] = set()
    for ancestor in (real.parent, *real.parents):
        candidates = [
            ancestor / "node_modules" / "@deepseek-ai" / "dsh-typert-protocol" / "package.json",
            ancestor / "packages" / "typert" / "protocol" / "package.json",
            *ancestor.glob(
                "node_modules/.pnpm/@deepseek-ai+dsh-typert-protocol@*/node_modules/"
                "@deepseek-ai/dsh-typert-protocol/package.json"
            ),
        ]
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate.parent
    return None


def codec_face(package: Path | None) -> str | None:
    """The strict codec face a protocol package declares: ``create`` or ``schema``."""
    if package is None:
        return None
    for relative in ("lib/types/types.d.ts", "src/types.ts"):
        declaration = package / relative
        if not declaration.exists():
            continue
        text = declaration.read_text(encoding="utf-8", errors="replace")
        for face, pattern in _FACE_PATTERNS:
            if pattern.search(text):
                return face
    return None


def inspect(label: str, launcher: Path) -> Harness:
    """Everything the comparison needs about one harness."""
    return Harness(
        label=label,
        launcher=launcher,
        version=_version(launcher),
        codec_face=codec_face(protocol_package(launcher)),
    )


def disagreement(pinned: Harness, serving: Harness) -> str | None:
    """Why the two harnesses cannot both be the profile's contract, or None.

    A version difference is the primary disagreement. When the versions match —
    the ``0.1.6-alpha.1`` fork case — a differing strict codec face still names a
    real divergence, because the two trees are not the published release the pin
    refers to.
    """
    if pinned.launcher.resolve() == serving.launcher.resolve():
        return None
    if pinned.version is None:
        return f"the pinned harness at {pinned.launcher} did not answer `--version`"
    if serving.version is None:
        return f"the serving dsh at {serving.launcher} did not answer `--version`"
    if pinned.version != serving.version:
        return (
            f"the {pinned.describe()} and the {serving.describe()} "
            "are different releases"
        )
    if (
        pinned.codec_face is not None
        and serving.codec_face is not None
        and pinned.codec_face != serving.codec_face
    ):
        return (
            f"the {pinned.describe()} and the {serving.describe()} report the "
            "same version but declare different strict TypertCodec contracts"
        )
    return None


def check() -> HarnessCheck:
    """Inspect the pinned harness and the harness that serves the profile.

    A machine with no separate ``dsh`` on ``PATH`` is green: there is no second
    harness to disagree with, which is the plain-clone and CI case.
    """
    pinned = inspect("pinned", pinned_launcher())
    launcher = serving_launcher()
    if launcher is None:
        return HarnessCheck(pinned=pinned, serving=None, reason=None)
    serving = inspect("serving", launcher)
    return HarnessCheck(pinned=pinned, serving=serving, reason=disagreement(pinned, serving))


def main(argv: list[str] | None = None) -> int:
    """Print the comparison and exit nonzero on a disagreement.

    The argument list is accepted for ``python -m`` symmetry and ignored; the
    check reads only the environment and the checkout.
    """
    del argv
    outcome = check()
    print(f"  {outcome.pinned.describe()}")
    if outcome.serving is None:
        print(
            f"  no separate dsh on PATH ({SERVING_DSH_ENV} unset or empty); "
            "the pinned launcher is the serving harness"
        )
    else:
        print(f"  {outcome.serving.describe()}")
    if outcome.reason is not None:
        print(f"harness check: {outcome.reason}")
        return 1
    print("harness check: the pinned harness and the serving dsh agree")
    return 0


if __name__ == "__main__":  # pragma: no cover - the module entry point
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "SERVING_DSH_ENV",
    "Harness",
    "HarnessCheck",
    "check",
    "codec_face",
    "disagreement",
    "inspect",
    "main",
    "pinned_launcher",
    "protocol_package",
    "serving_launcher",
]
