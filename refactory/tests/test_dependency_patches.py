"""Every declared dependency patch is well-formed and actually applied.

A patch with a stale hunk count is rejected by `git apply`, and pnpm can then install the
package *unpatched* while its virtual-store path and the lockfile both carry the patch
hash — so the build and the tests run against code that is not patched at all. F1's
`dsh-subagent-claude-code` patch hit exactly that on 2026-10-29: two headers said +9 and
+15 where the body had +10 and +20, and the installed file had none of the patch's lines.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
#: Tokens a patch can introduce: a quoted string, or a name. Kept deliberately broad; what
#: makes one a marker is that the patch's own context and removed lines do not contain it.
_TOKEN = re.compile(r'"[^"\n]{6,}"|`[^`\n]{6,}`|\b[A-Za-z_][A-Za-z0-9_]{5,}\b')
#: Files a patch may change, by suffix. Sourcemaps are not read.
_SUFFIXES = {".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".json"}


def _declared() -> dict[str, Path]:
    """Each patched dependency and the patch file the workspace declares for it."""
    data = yaml.safe_load((REPO / "pnpm-workspace.yaml").read_text()) or {}
    return {name: REPO / relative for name, relative in (data.get("patchedDependencies") or {}).items()}


def _hunks(text: str) -> list[tuple[int, int, int, int, int]]:
    """(line number, header -old, body -old, header +new, body +new) for each hunk."""
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        match = _HUNK.match(line)
        if not match:
            continue
        old, new, old_body, new_body = int(match.group(2) or 1), int(match.group(4) or 1), 0, 0
        for body in lines[index + 1:]:
            if body.startswith("@@"):
                break
            if body.startswith("\\"):
                continue
            if not body or body[0] not in " +-":
                break
            old_body += body[0] in " -"
            new_body += body[0] in " +"
        found.append((index + 1, old, old_body, new, new_body))
    return found


def _introduced(text: str) -> list[str]:
    """Tokens the patch adds and does not otherwise contain: present only once applied.

    A token that already sits in a context or removed line is in the package before the
    patch runs, so finding it in the installed copy proves nothing; those are excluded.
    """
    added = "\n".join(line[1:] for line in text.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    elsewhere = "\n".join(line[1:] for line in text.splitlines()
                          if line.startswith((" ", "-")) and not line.startswith("---"))
    present = set(_TOKEN.findall(elsewhere))
    markers: list[str] = []
    for token in _TOKEN.findall(added):
        if token in present or token in markers:
            continue
        markers.append(token)
    return markers


@pytest.mark.parametrize("name", sorted(_declared()))
def test_the_patch_is_well_formed(name):
    path = _declared()[name]
    assert path.is_file(), f"{name} declares a patch that does not exist: {path}"
    text = path.read_text()
    bad = [hunk for hunk in _hunks(text) if (hunk[1], hunk[3]) != (hunk[2], hunk[4])]
    assert not bad, (
        f"{path.name} has a hunk whose header does not match its body "
        f"(line, -header, -body, +header, +body): {bad}. `git apply` rejects it, and pnpm can "
        "then leave the package unpatched while recording the patch hash")


def _installed(package: str) -> Path | None:
    """The copy the build actually resolves, or None when it is not installed.

    The declared key is ``@scope/name@version``; the link under node_modules has no version.
    ``node_modules/<name>`` is what a build follows, and the ``.pnpm`` directory behind it
    carries the patch hash. Falls back to a single virtual-store copy, and refuses to guess
    between several.
    """
    name, _, version = package.rpartition("@")
    if not name:
        return None
    scope, _, bare = name.rpartition("/")
    linked = REPO / "node_modules" / scope / bare if scope else REPO / "node_modules" / bare
    if linked.is_dir():
        return linked
    directory = scope.replace("/", "+") if scope else ""
    pattern = f"{directory}+{bare}@{version}" if directory else f"{bare}@{version}"
    candidates = [path / "node_modules" / scope / bare if scope else path / "node_modules" / bare
                  for path in (REPO / "node_modules/.pnpm").glob(pattern + "*")]
    existing = sorted(path for path in candidates if path.is_dir())
    return existing[0] if len(existing) == 1 else None


@pytest.mark.parametrize("name", sorted(_declared()))
def test_the_patch_is_applied(name):
    """A patch pnpm recorded but did not apply leaves the package unpatched, silently."""
    directory = _installed(name)
    if directory is None:
        pytest.skip("node_modules is not installed, so what pnpm did cannot be inspected")
    markers = _introduced(_declared()[name].read_text())
    assert markers, (
        f"{_declared()[name].name} introduces no token absent from its own context, so it "
        "cannot be checked for having been applied; say so here rather than passing silently")
    text = "\n".join(path.read_text(errors="ignore") for path in directory.rglob("*")
                     if path.suffix in _SUFFIXES and path.is_file())
    missing = [marker for marker in markers if marker.strip('"`') not in text]
    assert not missing, (
        f"{name} is installed at {directory} but {len(missing)} of the {len(markers)} tokens its "
        f"patch introduces are absent ({', '.join(missing[:5])}): pnpm recorded the patch and did "
        "not apply it, so the build is running unpatched code. Reinstall with `pnpm install "
        "--force` after fixing the patch")
