"""Tests for versions derived from upstream and custom-suffixed tags."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.version import InvalidVersion

from _versioning import SuffixedVersion


@pytest.mark.parametrize(
    ("tag_version", "package_version"),
    [
        ("0.4.1", "0.4.1"),
        ("0.4.1-rc1", "0.4.1rc1"),
        ("0.4.1-rhlw", "0.4.1+rhlw"),
        ("0.4.1-rhlw.2", "0.4.1+rhlw.2"),
    ],
)
def test_tag_version_conversion(tag_version: str, package_version: str) -> None:
    """Keep standard versions and normalize custom release suffixes."""
    assert str(SuffixedVersion(tag_version)) == package_version


@pytest.mark.parametrize("tag_version", ["not-a-version", "0.4.1-bad!", "0.4.1-"])
def test_invalid_tag_version_is_rejected(tag_version: str) -> None:
    """Reject invalid tag versions instead of publishing misleading metadata."""
    with pytest.raises(InvalidVersion):
        SuffixedVersion(tag_version)


@pytest.mark.parametrize(
    ("tag", "package_version"),
    [("v0.4.1", "0.4.1"), ("v0.4.1-team", "0.4.1+team")],
)
def test_tag_builds_wheel_with_expected_version(
    tmp_path: Path, tag: str, package_version: str
) -> None:
    """Build clean tagged checkouts through the configured PEP 517 backend."""
    project = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    package = checkout / "v1" / "factory"
    package.mkdir(parents=True)
    for name in ("pyproject.toml", "build_backend.py", "_versioning.py"):
        shutil.copyfile(project / name, checkout / "v1" / name)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (checkout / ".gitignore").write_text("v1/factory/_version.py\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Test"),
        ("git", "config", "user.email", "test@example.com"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "Test release"),
        ("git", "tag", tag),
    ):
        subprocess.run(command, cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        (
            "uv", "build", "--offline", "--no-build-isolation", "--python", sys.executable,
            "--wheel", "v1",
        ),
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    wheel = next((checkout / "v1" / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith("/METADATA"))
        metadata = archive.read(metadata_name)
        generated = archive.read("factory/_version.py")
    assert f"Version: {package_version}\n".encode() in metadata
    assert f"__version__ = version = '{package_version}'".encode() in generated
