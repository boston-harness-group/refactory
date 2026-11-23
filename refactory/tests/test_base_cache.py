"""Tests for the base-check cache guards.

Covers _read_cache rejection of d4b-shaped entries, _try_base not caching
unfinished bases, and valid cache acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from refactory_dsh.base_check import (
    CACHE_FILE,
    BaseCheckResult,
    _read_cache,
)


@pytest.fixture
def cache_dir(tmp_path):
    """Create the .refactory directory for cache files."""
    cache_path = tmp_path / CACHE_FILE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_cache_file(cache_dir: Path, data: dict) -> None:
    """Write a cache file directly."""
    (cache_dir / CACHE_FILE).write_text(json.dumps(data, indent=2) + "\n")


def test_d4b_cache_rejected(cache_dir):
    """A d4b-shaped entry (ok=false, failures=[], environment_failure=null) is rejected."""
    data = {
        "ok": False,
        "failures": [],
        "environment_failure": None,
        "base_clean": True,
        "cache_key": "test-key",
        "output": "partial output...",
        "exit_code": 124,
        "base_complete": False,
    }
    _write_cache_file(cache_dir, data)
    result = _read_cache(cache_dir, "test-key")
    assert result is None, "d4b-shaped cache entry should be rejected"


def test_canary_repair_cache_rejected(cache_dir):
    """A canary-repair-shaped entry (exit 1, no failures) is rejected."""
    data = {
        "ok": False,
        "failures": [],
        "environment_failure": None,
        "base_clean": True,
        "cache_key": "test-key",
        "output": "credential guard failed",
        "exit_code": 1,
    }
    _write_cache_file(cache_dir, data)
    result = _read_cache(cache_dir, "test-key")
    assert result is None, "canary-repair-shaped cache entry should be rejected"


def test_valid_failure_cache_accepted(cache_dir):
    """A cache entry with actual failures is accepted."""
    data = {
        "ok": False,
        "failures": ["tests/test_a.py::test_x"],
        "environment_failure": None,
        "base_clean": True,
        "cache_key": "test-key",
        "output": "FAILED tests/test_a.py::test_x",
        "exit_code": 1,
    }
    _write_cache_file(cache_dir, data)
    result = _read_cache(cache_dir, "test-key")
    assert result is not None
    assert "tests/test_a.py::test_x" in result.failures


def test_valid_pass_cache_accepted(cache_dir):
    """A cache entry with ok=True is accepted."""
    data = {
        "ok": True,
        "failures": [],
        "environment_failure": None,
        "base_clean": True,
        "cache_key": "test-key",
        "output": "all passed",
        "exit_code": 0,
    }
    _write_cache_file(cache_dir, data)
    result = _read_cache(cache_dir, "test-key")
    assert result is not None
    assert result.ok is True


def test_environment_failure_cache_accepted(cache_dir):
    """A cache entry with environment_failure set is accepted (deterministic env failure)."""
    data = {
        "ok": False,
        "failures": [],
        "environment_failure": "pip install failed",
        "base_clean": True,
        "cache_key": "test-key",
        "output": "pip install error",
        "exit_code": 1,
    }
    _write_cache_file(cache_dir, data)
    result = _read_cache(cache_dir, "test-key")
    assert result is not None
    assert result.environment_failure == "pip install failed"


def test_wrong_key_returns_none(cache_dir):
    """A cache entry with a different key returns None."""
    data = {
        "ok": True,
        "failures": [],
        "base_clean": True,
        "cache_key": "other-key",
    }
    _write_cache_file(cache_dir, data)
    result = _read_cache(cache_dir, "test-key")
    assert result is None
