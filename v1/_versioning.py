"""Normalize custom release tag suffixes for Python package metadata."""

import re

from packaging.version import InvalidVersion, Version


_SUFFIXED_RELEASE = re.compile(
    r"(?P<release>[0-9]+\.[0-9]+\.[0-9]+)-(?P<suffix>[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*)\Z"
)


class SuffixedVersion(Version):
    """Use PEP 440 local versions for otherwise invalid custom tag suffixes."""

    def __init__(self, version: str) -> None:
        """Parse standard versions unchanged and map custom suffixes to local versions."""
        try:
            super().__init__(version)
        except InvalidVersion:
            match = _SUFFIXED_RELEASE.fullmatch(version)
            if match is None:
                raise
            super().__init__(f"{match['release']}+{match['suffix']}")
