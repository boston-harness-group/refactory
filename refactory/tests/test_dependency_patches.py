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
PATCHES = REPO / "patches"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
#: A string literal or a name this repository's patches introduce: distinctive enough that
#: finding it in the installed package means the patch is applied.
_MARKER = re.compile(r'"[^"\n]{6,}"|_rf[A-Za-z0-9]+')


def _declared() -> dict[str, Path]:
    """Each patched dependency and the patch file the workspace declares for it."""
    workspace = REPO / "pnpm-workspace.yaml"
    data = yaml.safe_load(workspace.read_text()) or {}
    declared = data.get("patchedDependencies") or {}
    return {name: REPO / relative for name, relative in declared.items()}


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


def _markers(text: str) -> list[str]:
    """Distinctive strings the patch's added lines introduce."""
    added = "\n".join(line[1:] for line in text.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    seen: list[str] = []
    for match in _MARKER.findall(added):
        if match not in seen:
            seen.append(match)
    return seen


@pytest.mark.parametrize("name", sorted(_declared()))
def test_the_patch_is_well_formed(name):
    path = _declared()[name]
    assert path.is_file(), f"{name} declares a patch that does not exist: {path}"
    text = path.read_text()
    bad = [(line, old, old_body, new, new_body) for line, old, old_body, new, new_body in _hunks(text)
           if (old, new) != (old_body, new_body)]
    assert not bad, (
        f"{path.name} has a hunk whose header does not match its body "
        f"(line, -header, -body, +header, +body): {bad}. `git apply` rejects it, and pnpm can "
        "then leave the package unpatched while recording the patch hash")


def _installed(package: str) -> Path | None:
    """The installed directory for a patched dependency, or None when it is not installed."""
    parts = package.rsplit("@", 1)[0].split("/")
    scope, bare = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
    pattern = f"{scope}+{bare}@*" if scope else f"{bare}@*"
    candidates = list((REPO / "node_modules/.pnpm").glob(pattern))
    for candidate in candidates:
        directory = candidate / "node_modules" / scope / bare if scope else candidate / "node_modules" / bare
        if directory.is_dir():
            return directory
    return None


@pytest.mark.parametrize("name", sorted(_declared()))
def test_the_patch_is_applied(name):
    """A patch pnpm recorded but did not apply leaves the package unpatched, silently."""
    directory = _installed(name)
    if directory is None:
        pytest.skip("node_modules is not installed, so what pnpm did cannot be inspected")
    markers = _markers(_declared()[name].read_text())
    if not markers:
        pytest.skip("the patch introduces no distinctive string to look for")
    text = "\n".join(path.read_text(errors="ignore") for path in directory.rglob("*.js"))
    assert any(marker.strip('"') in text for marker in markers), (
        f"{name} is installed at {directory} but none of the patch's markers "
        f"({', '.join(markers[:4])}) is in it: pnpm recorded the patch and did not apply it. "
        "Reinstall with `pnpm install --force` after fixing the patch")
